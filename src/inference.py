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


def calc_coordicate(pr: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
    """calculate pred-bbox coordinate

    Args:
        pr (torch.Tensor): (N, P, 4)
        df (torch.Tensor): (P, 4) -> (1, P, 4)

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

    return torch.stack([p_cx, p_cy, p_w, p_h], dim=2)


def calc_score(pr: torch.Tensor) -> torch.Tensor:
    """calculate pred-bbox score

    Args:
        pr (torch.Tensor): (N, P, C)

    Returns:
        torch.Tensor (N, P, C-4): class score
    """
    _, _, C = pr.shape

    max_mask = torch.eye(C - 4)[torch.max(pr[:, :, 4:], dim=2).indices].to(pr.device)
    return F.softmax(pr[:, :, 4:], dim=2) * max_mask


def calc_iou(t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """calculate iou

    Args:
        t (torch.Tensor): (N, T, 4) -> (N, T, 1, 4)
        s (torch.Tensor): (N, S, 4) -> (N, 1, S, 4)

    Returns:
        torch.Tensor: (N, T, S)
    """
    t = t.unsqueeze(2)
    s = s.unsqueeze(1)

    t_cx, t_cy, t_w, t_h = [t[:, :, :, i] for i in range(4)]
    s_cx, s_cy, s_w, s_h = [s[:, :, :, i] for i in range(4)]

    w = (torch.min(t_cx + t_w / 2, s_cx + s_w / 2) - torch.max(t_cx - t_w / 2, s_cx - s_w / 2)).clamp(min=0)
    h = (torch.min(t_cy + t_h / 2, s_cy + s_h / 2) - torch.max(t_cy - t_h / 2, s_cy - s_h / 2)).clamp(min=0)

    return torch.where(w * h > 0, w * h / (t_w * t_h + s_w * s_h - w * h), w * h)


def non_maximum_suppression(outputs: torch.Tensor, iou_thresh: float = 0.5) -> torch.Tensor:
    """execute non-maximum-suppression

    Args:
        outputs (torch.Tensor): (N, P, C)
        iou_thresh (float, optional): Remove predicted values ​​above this threshold. Defaults to 0.5.

    Returns:
        torch.Tensor: (N, P, C)
    """
    def nms(t: torch.Tensor) -> torch.Tensor:
        """suppress tensor

        Args:
            t (torch.Tensor): (N, P, C-4, P)

        Returns:
            torch.Tensor: (N, P, C-4)
        """
        suppressed_t = t[:, :, :, 0]
        for i in range(1, P):
            t[:, :, :, i] *= suppressed_t
            suppressed_t += t[:, :, :, i]
        return suppressed_t

    N, P, _ = outputs.shape

    select_flags = iou_thresh < calc_iou(outputs, outputs) + torch.eye(P).bool().to(outputs.device)
    orders = torch.sort(outputs[:, :, 4:], dim=1, descending=True).indices
    valid_mask = nms(t=torch.stack([select_flags[n][orders[n]] for n in range(N)]))
    outputs[:, :, 4:] = outputs[:, :, 4:] * valid_mask

    return outputs


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
                        # case: void or noise
                        if class_id == 0:
                            continue

                        # calc coord
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
