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
            torch.Tensor: loss
        """
        # constant definition
        N = pred_bboxes.shape[0]
        P = pred_bboxes.shape[1]
        C = pred_bboxes.shape[2] - 4

        # matching
        is_match = self.match(gt=gt_bboxes, df=default_bboxes)

        # localization loss
        l = pred_bboxes[:, :, :4].unsqueeze(2)
        g = self.calc_delta(gt=gt_bboxes, df=default_bboxes)
        l_loc = (self.smooth_l1(l - g) * is_match).sum(dim=2)

        # confidence loss
        # positive
        softmax_pos = self.softmax_cross_entropy(pr=pred_bboxes[:, :, 4:], gt=gt_bboxes[:, :, 4:])
        l_conf = (softmax_pos * is_match).sum(dim=2)

        # negative
        gt_neg = torch.eye(C)[0].unsqueeze(0).unsqueeze(1)
        softmax_neg = self.softmax_cross_entropy(pr=pred_bboxes[:, :, 4:], gt=gt_neg)
        l_conf += (softmax_neg.squeeze() * ((is_match.sum(dim=2) == 0) * (-1)))

        # hard negative mining
        pos_num = (is_match.sum(dim=2) != 0).sum(dim=1)
        neg_num = P - pos_num
        pos_num, neg_num = self.split_pos_neg(pos_num, neg_num)

        valid_mask = torch.stack([torch.kthvalue(l_conf[i], k=neg_num[i]).values for i in range(N)]).unsqueeze(1) > l_conf
        valid_mask += -torch.stack([torch.kthvalue(-l_conf[i], k=pos_num[i]).values for i in range(N)]).unsqueeze(1) < l_conf

        # calculate loss
        loss = (((l_loc + a * l_conf.abs()) * valid_mask).sum(dim=1) / pos_num).mean()

        return loss

    def match(self, gt: torch.Tensor, df: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
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

    def calc_delta(self, gt: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
        """calculate g-hat

        Args:
            gt (torch.Tensor): (N, G, 4) -> (N, 1, G, 4)
            df (torch.Tensor): (1, P, 1, 4)

        Returns:
            torch.Tensor: g-hat tensor
        """
        gt = gt.unsqueeze(1)
        df = df.unsqueeze(0).unsqueeze(2)

        g_cx, g_cy, g_w, g_h = gt[:, :, :, 0], gt[:, :, :, 1], gt[:, :, :, 2], gt[:, :, :, 3]
        d_cx, d_cy, d_w, d_h = df[:, :, :, 0], df[:, :, :, 1], df[:, :, :, 2], df[:, :, :, 3]
        g_cx = (g_cx - d_cx) / d_w
        g_cy = (g_cy - d_cy) / d_h
        g_w = torch.log(g_w / d_w)
        g_h = torch.log(g_h / d_h)

        return torch.stack([g_cx, g_cy, g_w, g_h], dim=3)

    def smooth_l1(self, x: torch.Tensor) -> torch.Tensor:
        """calculate smooth l1

        Args:
            x (torch.Tensor): any tensor

        Returns:
            torch.Tensor: smooth l1
        """
        mask = x.abs() < 1
        return (0.5 * x ** 2 * mask + (x.abs() - 0.5) * (~mask)).sum(dim=3)

    def softmax_cross_entropy(self, pr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """calculate softmax cross-entropy

        Args:
            pr (torch.Tensor): (N, P, class_num) -> (N, P, 1, class_num)
            gt (torch.Tensor): (N, G, class_num) -> (N, 1, G, class_num)

        Returns:
            torch.Tensor: softmax cross-entropy
        """
        pr = pr.unsqueeze(2)
        gt = gt.unsqueeze(1)

        sm = torch.exp(pr) / torch.exp(pr).sum(dim=3, keepdims=True)
        return -(gt * torch.log(sm)).sum(dim=3)

    def split_pos_neg(self, pos_num: torch.Tensor, neg_num: torch.Tensor) -> torch.Tensor:
        """split pos:neg = 1:3

        Args:
            pos_num (torch.Tensor): (N)
            neg_num (torch.Tensor): (N)

        Returns:
            torch.Tensor: (N)
        """
        cond = pos_num * 3 > neg_num
        return torch.where(cond, neg_num // 3, pos_num), torch.where(cond, neg_num, pos_num * 3)
