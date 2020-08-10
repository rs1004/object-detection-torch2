from typing import Tuple
from dataset import PascalVOCDataset
from model import SSD
from augmentation import Compose, ToTensor
from utils import collate_fn
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
import argparse


def calc_bbox_location(pr: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
    """calculate bbox location

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

    # (cx, cy, w, h) => (xmin, ymin, xmax, ymax)
    p_x_min = p_cx - p_w / 2
    p_y_min = p_cy - p_h / 2
    p_x_max = p_cx + p_w / 2
    p_y_max = p_cy + p_h / 2

    return torch.stack([p_x_min, p_y_min, p_x_max, p_y_max], dim=2)


def calc_iou(pr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """calculate iou

    Args:
        pr (torch.Tensor): (N, P, 4) -> (N, P, 1, 4)
        gt (torch.Tensor): (N, G, 4) -> (N, 1, G, 4)

    Returns:
        torch.Tensor: (N, P, G)
    """
    pr = pr.unsqueeze(2)
    gt = gt.unsqueeze(1)

    p_cx, p_cy, p_w, p_h = [pr[:, :, :, i] for i in range(4)]
    g_cx, g_cy, g_w, g_h = [gt[:, :, :, i] for i in range(4)]

    w = (torch.min(p_cx + p_w / 2, g_cx + g_w / 2) - torch.max(p_cx - p_w / 2, g_cx - g_w / 2)).clamp(min=0)
    h = (torch.min(p_cy + p_h / 2, g_cy + g_h / 2) - torch.max(p_cy - p_h / 2, g_cy - g_h / 2)).clamp(min=0)

    return torch.where(w * h > 0, w * h / (p_w * p_h + g_w * g_h - w * h), w * h)


def get_max_iou(iou: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """get max iou tensor

    Args:
        iou (torch.Tensor): (N, P, G) -> (N, P, G, 1)
        gt (torch.Tensor): (N, G, C) -> (N, 1, G, C)

    Returns:
        torch.Tensor: (N, P, C)
    """
    iou = iou.unsqueeze(3)
    gt = gt.unsqueeze(1)

    return (iou * gt).max(dim=2).values


def calc_indicators(max_iou: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5, score_threshold: float = 0.5
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """calculate precision, recall, and accuracy

    Args:
        max_iou (torch.Tensor): (N, P, C)
        scores (torch.Tensor): (N, P, C)
        iou_threshold (float, optional): threshold that bbox matches. Defaults to 0.5.
        score_threshold (float, optional): threshold that bbox predicts. Defaults to 0.5.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (precision, recall, accuracy)
    """
    tp = ((max_iou >= iou_threshold) * (scores >= score_threshold)).sum(dim=1).float()
    fp = ((max_iou < iou_threshold) * (scores >= score_threshold)).sum(dim=1).float()
    fn = ((max_iou >= iou_threshold) * (scores < score_threshold)).sum(dim=1).float()
    tn = ((max_iou < iou_threshold) * (scores < score_threshold)).sum(dim=1).float()

    precision = torch.where(tp + fp > 0, tp / (tp + fp), tp + fp)
    recall = torch.where(tp + fn > 0, tp / (tp + fn), tp + fn)
    accuracy = torch.where(tp + fp + fn + tn > 0, (tp + tn) / (tp + fp + fn + tn), tp + fp + fn + tn)

    return precision, recall, accuracy


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

    precision = torch.empty((0, dataset.num_classes), device=device)
    recall = torch.empty((0, dataset.num_classes), device=device)
    accuracy = torch.empty((0, dataset.num_classes), device=device)
    with torch.no_grad():
        with tqdm(dataloader, total=len(dataloader)) as pbar:
            for images, gts in pbar:
                # description
                pbar.set_description('[Evaluate]')

                # to GPU device
                images = images.to(device)
                gts = gts.to(device)

                # evaluate
                outputs = net(images)
                locs = calc_bbox_location(pr=outputs[:, :, :4], df=defaults)
                iou = calc_iou(pr=locs, gt=gts[:, :, :4])
                max_iou = get_max_iou(iou=iou, gt=gts[:, :, 5:])
                pre, rec, acc = calc_indicators(max_iou=max_iou, scores=F.softmax(outputs[:, :, 5:], dim=2))
                precision = torch.cat([precision, pre], dim=0)
                recall = torch.cat([recall, rec], dim=0)
                accuracy = torch.cat([accuracy, acc], dim=0)
        average_precision = precision.mean(dim=0)
        average_recall = recall.mean(dim=0)
        average_accuracy = accuracy.mean(dim=0)

        print(f'mAP: {average_precision.mean()}, mAR: {average_recall.mean()}, mACC: {average_accuracy.mean()}')

    print('Finished Evaluate')
