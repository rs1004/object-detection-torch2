import torch
import torch.nn as nn


class SSDLoss(nn.Module):
    def __init__(self):
        # initialize
        super(SSDLoss, self).__init__()

    def forward(self, outputs, default_bboxes, gt_bboxes, a=1):
        '''
        Args
            outputs: torch.tensor (size: [batch_size, bbox_num, class_num + 4])
            default_bboxes: torch.tensor (size: [bbox_num, 4])
            gt_bboxes: list: ([{'c': class_id, 'bbox': torch.tensor([cx, cy, w, h])}, …])
        '''
        loss = 0.0
        for pred_bboxes in outputs:
            N = 0
            l_loc = torch.empty(0)
            l_conf_pos = torch.empty(0)
            l_conf_neg = torch.empty(0)
            for i in range(pred_bboxes.shape[0]):
                is_negative = True
                for j in range(len(gt_bboxes)):
                    jaccard_overlap = calc_iou(default_bboxes[i], gt_bboxes[j]['bbox'])
                    if jaccard_overlap > 0.5:
                        l = pred_bboxes[i][:4]
                        g = calc_delta(gt_bboxes[j]['bbox'], default_bboxes[i])
                        l_loc = torch.cat([l_loc, smooth_l1(l - g)])
                        l_conf_pos = torch.cat([l_conf_pos, -log_softmax(pred_bboxes[i][4:], gt_bboxes[j]['c'])])
                        N += 1
                        is_negative = False
                if is_negative:
                    l_conf_neg = torch.cat([l_conf_neg, -log_softmax(pred_bboxes[i][4:], 4)])

            pn, nn = split_pos_neg(len(l_conf_pos), len(l_conf_neg))
            p_indices = l_conf_pos.sort(descending=True).indices[:pn]
            n_indices = l_conf_neg.sort(descending=True).indices[:nn]
            loss_ = (l_loc[p_indices].sum() + a * l_conf_pos[p_indices].sum()) / N
            loss_ += a * l_conf_neg[n_indices].sum() / N
            loss += loss_ / outputs.shape[0]
        return loss


def calc_iou(bbox1, bbox2):
    cx1, cy1, w1, h1 = bbox1
    cx2, cy2, w2, h2 = bbox2
    w = torch.min(cx1 + w1/2, cx2 + w2/2) - torch.max(cx1 - w1/2, cx2 - w2/2)
    h = torch.min(cy1 + h1/2, cy2 + h2/2) - torch.max(cy1 - h1/2, cy2 - h2/2)
    if w > 0 and h > 0:
        return w * h / (w1 * h1 + w2 * h2 - w * h)
    else:
        return torch.tensor(0.)


def calc_delta(gt_bbox, default_bbox):
    g_cx, g_cy, g_w, g_h = gt_bbox
    d_cx, d_cy, d_w, d_h = default_bbox
    g_cx = (g_cx - d_cx) / d_w
    g_cy = (g_cy - d_cy) / d_h
    g_w = torch.log(g_w / d_w)
    g_h = torch.log(g_h / d_h)
    return torch.tensor([g_cx, g_cy, g_w, g_h])


def smooth_l1(x):
    mask = torch.abs(x) < 1
    return torch.sum(0.5 * x ** 2 * mask + (torch.abs(x) - 0.5) * (~mask)).unsqueeze(0)


def log_softmax(y, t):
    return torch.log(torch.exp(y) / torch.exp(y).sum())[[t]]


def split_pos_neg(pos_num, neg_num):
    if pos_num * 3 > neg_num:
        return neg_num // 3, neg_num
    else:
        return pos_num, pos_num * 3
