from dataset import PascalVOCDataset
from model import SSD
from augmentation import Compose, ToTensor
from utils import collate_fn, calc_coordicate, calc_score, non_maximum_suppression, LabelMap
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
import seaborn as sns
import torch
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imsize', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=2)
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

                outputs[:, :, :4] = calc_coordicate(pr=outputs, df=defaults)
                outputs[:, :, 4:] = calc_score(pr=outputs)
                outputs = non_maximum_suppression(outputs=outputs)

                bbox_locs = outputs[:, :, :4]
                bbox_confs = outputs[:, :, 4:]
                for i in range(len(images)):
                    image = Image.fromarray((images[i].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
                    draw = ImageDraw.Draw(image)
                    locs = bbox_locs[i]
                    labels = torch.max(bbox_confs[i], dim=1)
                    for loc, class_id, score in zip(locs, labels.indices, labels.values):
                        # case: void
                        if class_id == 0:
                            continue

                        # calc location
                        cx, cy, w, h = [_.item() for _ in loc * args.imsize]
                        xmin, ymin, xmax, ymax = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
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
