from dataset import PascalVOCDataset
from model import SSD
from augmentation import Compose, ToTensor
from utils import collate_fn, LabelMap
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
import seaborn as sns
import torch
import torch.nn.functional as F
import argparse


def calc_bbox_location(pr: torch.Tensor, df: torch.Tensor, imsize: int) -> torch.Tensor:
    """calculate bbox location

    Args:
        pr (torch.Tensor): (N, P, 4)
        df (torch.Tensor): (P, 4) -> (1, P, 4)
        imsize (int): image size

    Returns:
        torch.Tensor (N, P, 4): location coordinate
    """
    df = df.unsqueeze(0)

    p_cx, p_cy, p_w, p_h = [pr[:, :, i] for i in range(4)]
    d_cx, d_cy, d_w, d_h = [df[:, :, i] for i in range(4)]

    # Δ(cx, cy, w, h) => (cx, cy, w, h)
    p_cx = d_w * p_cx + d_cx
    p_cy = d_h * p_cy + d_cy
    p_w = d_w * torch.exp(p_w)
    p_h = d_h * torch.exp(p_h)

    # (cx, cy, w, h) => (xmin, ymin, xmax, ymax)
    p_x_min = p_cx - p_w / 2
    p_y_min = p_cy - p_h / 2
    p_x_max = p_cx + p_w / 2
    p_y_max = p_cy + p_h / 2

    return torch.stack([p_x_min, p_y_min, p_x_max, p_y_max], dim=2) * imsize


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imsize', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--weights', type=str, default='weights.pth')
    args = parser.parse_args()

    weights_path = Path(args.result_dir) / 'detection' / args.weights
    out_dir = Path(args.result_dir) / 'detection'
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = Compose([
        ToTensor()])

    dataset = PascalVOCDataset(
        purpose='detection',
        data_dirs='/work/data/VOCdevkit/VOC2007',
        data_list_file_name='test.txt',
        imsize=args.imsize,
        transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = SSD(
        num_classes=dataset.num_classes + 1,
        weights_path=weights_path,
        weights_path_vgg16=Path(args.result_dir) / 'classification' / args.weights
    )
    net.to(device)
    defaults = net.default_bboxes.to(device)

    current_palette = sns.color_palette('hls', n_colors=dataset.num_classes + 1)
    labelmap = LabelMap('PascalVOC')

    n = 1
    with torch.no_grad():
        with tqdm(dataloader, total=len(dataloader)) as pbar:
            for images, _ in pbar:
                # description
                pbar.set_description('[Inference]')

                # to GPU device
                images = images.to(device)

                # generate image
                outputs = net(images)
                bbox_locs = calc_bbox_location(pr=outputs, df=defaults, imsize=args.imsize)
                bbox_confs = outputs[:, :, 4:]
                for i in range(args.batch_size):
                    image = Image.fromarray((images[i].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
                    draw = ImageDraw.Draw(image)
                    locs = bbox_locs[i].cpu()
                    labels = torch.max(F.softmax(bbox_confs[i], dim=1).cpu(), dim=1)
                    for loc, class_id, score in zip(locs, labels.indices, labels.values):
                        # case: void or noise
                        if (class_id == 0) or (score < 0.5):
                            continue

                        # calc coord
                        xmin, ymin, xmax, ymax = [_.item() for _ in loc]
                        left_top = (max(xmin, 0), max(ymin, 0))
                        right_bottom = (min(xmax, args.imsize), min(ymax, args.imsize))

                        # set text
                        text = f' {labelmap.id2name(class_id.item()-1)} {str(round(score.item(), 3))}'
                        text_loc = (max(xmin, 0), max(ymin, 0) - 11)
                        text_back_loc = (max(xmin, 0) + len(text) * 6, max(ymin, 0))

                        # draw bbox
                        color = tuple(int(c * 255) for c in current_palette[class_id])
                        draw.rectangle(left_top + right_bottom, outline=color)
                        draw.rectangle(text_loc + text_back_loc, fill=color, outline=color)
                        draw.text(text_loc, text, fill=(0, 0, 0, 0))

                    image.save(out_dir / f'{n:06}.png')
                    n += 1

    print('Finished Inference')
