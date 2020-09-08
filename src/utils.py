import json
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path


def collate_fn(batch):
    images = []
    gts = []
    for image, gt in batch:
        images.append(image)
        gts.append(gt)
    images = torch.stack(images, dim=0)
    gts = pad_sequence(gts, batch_first=True)
    return images, gts


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


class LabelMap:
    def __init__(self, ds_name):
        self.ds_name = ds_name
        self.labels = self._get_labels()

    def _get_labels(self):
        label_map_path = Path(__file__).parent / 'labelmap.json'
        with open(label_map_path, 'r') as f:
            labels = json.load(f)[self.ds_name]
        return labels

    def __len__(self):
        return len(self.labels)

    def name2id(self, name):
        return self.labels.index(name)

    def id2name(self, id):
        return self.labels[id]
