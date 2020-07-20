import torch
import torch.nn as nn


class SSDLoss(nn.Module):
    def __init__(self):
        # initialize
        super(SSDLoss, self).__init__()

    def forward(self, pred_bboxes: torch.Tensor, default_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, a: int = 1) -> torch.Tensor:
        """calculate loss

        Args:
            pred_bboxes (torch.Tensor)   : (N, P, 4 + C)
            default_bboxes (torch.Tensor): (P, 4)
            gt_bboxes (torch.Tensor)     : (N. G. 4 + C)
            a (int, optional): weight term of loss formula. Defaults to 1.

        Returns:
            torch.Tensor: [description]
        """
        # constant definition
        N = pred_bboxes.shape[0]
        P = pred_bboxes.shape[1]
        C = pred_bboxes.shape[2] - 4

        # matching
        is_match = match(gt=gt_bboxes, df=default_bboxes)

        # localization loss
        l = pred_bboxes[:, :, :4].unsqueeze(2)
        g = calc_delta(gt=gt_bboxes, df=default_bboxes)
        l_loc = (smooth_l1(l - g) * is_match).sum(dim=2)

        # confidence loss
        # positive
        softmax_pos = softmax_cross_entropy(pr=pred_bboxes[:, :, 4:], gt=gt_bboxes[:, :, 4:])
        l_conf = (softmax_pos * is_match).sum(dim=2)

        # negative
        gt_neg = torch.eye(C)[0].unsqueeze(0).unsqueeze(1)
        softmax_neg = softmax_cross_entropy(pr=pred_bboxes[:, :, 4:], gt=gt_neg)
        l_conf += (softmax_neg.squeeze() * ((is_match.sum(dim=2) == 0) * (-1)))

        # hard negative mining
        pos_num = (is_match.sum(dim=2) != 0).sum(dim=1)
        neg_num = P - pos_num
        pos_num, neg_num = split_pos_neg(pos_num, neg_num)

        valid_mask = torch.stack([torch.kthvalue(l_conf[i], k=neg_num[i]).values for i in range(N)]).unsqueeze(1) > l_conf
        valid_mask += -torch.stack([torch.kthvalue(-l_conf[i], k=pos_num[i]).values for i in range(N)]).unsqueeze(1) < l_conf

        # calculate loss
        loss = (((l_loc + a * l_conf.abs()) * valid_mask).sum(dim=1) / pos_num).mean()

        return loss


def match(gt: torch.Tensor, df: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """adapt matching strategy

    Args:
        gt (torch.Tensor): (N, G, 4) -> (N, 1, G, 4)
        df (torch.Tensor): (P, 4) -> (1, P, 1, 4)
        threshold (float, optional): threshold of iou. Defaults to 0.5.

    Returns:
        torch.Tensor: matching mask
    """
    gt = gt.unsqueeze(1)
    df = df.unsqueeze(0).unsqueeze(2)

    g_cx, g_cy, g_w, g_h = gt[:, :, :, 0], gt[:, :, :, 1], gt[:, :, :, 2], gt[:, :, :, 3]
    d_cx, d_cy, d_w, d_h = df[:, :, :, 0], df[:, :, :, 1], df[:, :, :, 2], df[:, :, :, 3]
    w = (torch.min(g_cx + g_w/2, d_cx + d_w/2) - torch.max(g_cx - g_w/2, d_cx - d_w/2)).clamp(min=0)
    h = (torch.min(g_cy + g_h/2, d_cy + d_h/2) - torch.max(g_cy - g_h/2, d_cy - d_h/2)).clamp(min=0)

    return (w * h / (g_w * g_h + d_w * d_h - w * h)) > threshold
