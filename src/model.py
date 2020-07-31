import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class VGG16(nn.Module):
    def __init__(self):
        # initialize
        super(VGG16, self).__init__()

        # feature extraction layer
        layers = []
        in_channels = 3
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M_P', 512, 512, 512, 'M', 512, 512, 512, 'M_P']
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'M_P':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=1)]
            else:
                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True)
                ]
                in_channels = v

        self.features = nn.Sequential(*layers)

        # classification layer
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )

        # load weights
        vgg16_bn = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16_bn', pretrained=True)
        self.load_state_dict(vgg16_bn.state_dict())

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def loss(self, output, target):
        output = F.softmax(output, dim=1)
        loss = nn.CrossEntropyLoss()(input=output, target=target)
        return loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class SSD(nn.Module):
    def __init__(self, num_classes, weights_path=None):
        # initialize
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.default_bboxes = self._get_default_bboxes()

        features = nn.ModuleDict()

        # vgg16 layer
        vgg16 = VGG16()
        num_table = {}
        name_map = {'Conv2d': 'conv', 'BatchNorm2d': 'bn', 'ReLU': 'act', 'MaxPool2d': 'pool'}
        layer_num = 1
        for m in vgg16.features:
            for param in m.parameters():
                param.requires_grad = False
            name = name_map[m._get_name()]
            if name in num_table:
                num_table[name] += 1
                features[f'{name}_{layer_num}_{num_table[name]}'] = m
            elif name == 'pool':
                if layer_num < 5:
                    features[f'{name}_{layer_num}'] = m
                layer_num += 1
                num_table.clear()
            else:
                num_table[name] = 1
                features[f'{name}_{layer_num}_{num_table[name]}'] = m

        # additional layer
        sub_num = 1
        in_channels = 512
        cfg = [(3, 1024, 1, 1), '.',                    # layer6
               (1, 1024, 1, 0), '.',                    # layer7
               (1, 256, 1, 0), (3, 512, 2, 1), '.',     # layer8
               (1, 128, 1, 0), (3, 256, 2, 1), '.',     # layer9
               (1, 128, 1, 0), (3, 256, 1, 0), '.',     # layer10
               (1, 128, 1, 0), (3, 256, 1, 0)]          # layer11
        for c in cfg:
            if c == '.':
                layer_num += 1
                sub_num = 1
            else:
                k, v, s, p = c
                features[f'conv_{layer_num}_{sub_num}'] = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=k, stride=s, padding=p)
                features[f'bn_{layer_num}_{sub_num}'] = nn.BatchNorm2d(v)
                features[f'act_{layer_num}_{sub_num}'] = nn.ReLU(inplace=True)

                sub_num += 1
                in_channels = v

        self.features = features

        self.classifier = nn.ModuleDict({
            'act_4_3': nn.Conv2d(in_channels=512, out_channels=4*(num_classes+4), kernel_size=3, padding=1),
            'act_7_1': nn.Conv2d(in_channels=1024, out_channels=6*(num_classes+4), kernel_size=3, padding=1),
            'act_8_2': nn.Conv2d(in_channels=512, out_channels=6*(num_classes+4), kernel_size=3, padding=1),
            'act_9_2': nn.Conv2d(in_channels=256, out_channels=6*(num_classes+4), kernel_size=3, padding=1),
            'act_10_2': nn.Conv2d(in_channels=256, out_channels=4*(num_classes+4), kernel_size=3, padding=1),
            'act_11_2': nn.Conv2d(in_channels=256, out_channels=4*(num_classes+4), kernel_size=3, padding=1),
        })

        # load weights
        if Path(weights_path).exists():
            self.load_state_dict(torch.load(weights_path))
        else:
            self._initialize_weights()

    def forward(self, x):
        batch_size = x.size(0)
        y = torch.empty((batch_size, 0, self.num_classes + 4))
        y = y.to(x.device)

        for name, layer in self.features.items():
            x = layer(x)
            if name in self.classifier:
                y = torch.cat([y, self.classifier[name](x).view(batch_size, -1, self.num_classes + 4)], dim=1)

        return y

    def _get_default_bboxes(self):
        def s_(k, m=6, s_min=0.2, s_max=0.9):
            return s_min + (s_max - s_min) * (k - 1) / (m - 1)

        default_bboxes = torch.empty(0, 4)
        cfg = [[38, 38, 4], [19, 19, 6], [10, 10, 6], [5, 5, 6], [3, 3, 4], [1, 1, 4]]

        for k, (m, n, l) in enumerate(cfg, start=1):
            aspects = [1, 2, 1/2, 'add'] if l == 4 else [1, 2, 3, 1/2, 1/3, 'add']
            for i in range(m):
                for j in range(n):
                    for a in aspects:
                        if a == 'add':
                            w = h = (s_(k) * s_(k+1)) ** 0.5
                        else:
                            w = s_(k) * (a ** 0.5)
                            h = s_(k) * ((1/a) ** 0.5)
                        new_bbox = torch.Tensor([[(i + 0.5) / m, (j + 0.5) / n, w, h]])
                        default_bboxes = torch.cat([default_bboxes, new_bbox])

        return default_bboxes

    def _initialize_weights(self):
        is_vgg = True
        for k, m in self.features.items():
            if k == 'conv_6_1':
                is_vgg = False
            if is_vgg:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def train_params(self) -> None:
        """generate params to train 

        Yields:
            nn.parameter.Parameter: trainable params
        """
        # generate features params
        is_vgg = True
        for name, layer in self.features.items():
            if name == 'conv_6_1':
                is_vgg = False
            if is_vgg:
                continue
            for param in layer.parameters():
                yield param

        # generate classifier params
        for _, layer in self.classifier.items():
            for param in layer.parameters():
                yield param

    def loss(self, pred_bboxes: torch.Tensor, default_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, a: int = 1) -> torch.Tensor:
        """calculate loss

        Args:
            pred_bboxes (torch.Tensor)   : (B, P, 4 + C)
            default_bboxes (torch.Tensor): (P, 4)
            gt_bboxes (torch.Tensor)     : (B. G. 4 + C)
            a (int, optional): weight term of loss formula. Defaults to 1.

        Returns:
            torch.Tensor: loss
        """
        # constant definition
        B = pred_bboxes.shape[0]
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
        gt_neg = torch.eye(C)[0].unsqueeze(0).unsqueeze(1).to(pred_bboxes.device)
        softmax_neg = self.softmax_cross_entropy(pr=pred_bboxes[:, :, 4:], gt=gt_neg)
        l_conf += (softmax_neg.squeeze() * ((is_match.sum(dim=2) == 0) * (-1)))

        # hard negative mining
        pos_num = (is_match.sum(dim=2) != 0).sum(dim=1)
        neg_num = P - pos_num
        pos_num, neg_num = self.split_pos_neg(pos_num, neg_num)

        valid_mask = torch.stack([self.kthvalue(l_conf[i], k=neg_num[i], mode='min') for i in range(B)]).unsqueeze(1) > l_conf
        valid_mask += -torch.stack([self.kthvalue(-l_conf[i], k=pos_num[i], mode='max') for i in range(B)]).unsqueeze(1) < l_conf

        # calculate loss (if pos_num = 0, then loss = 0)
        pos_num = torch.where(pos_num > 0, 1/pos_num, pos_num).float()
        loss = (((l_loc + a * l_conf.abs()) * valid_mask).sum(dim=1) * pos_num).mean()

        return loss

    def match(self, gt: torch.Tensor, df: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """adapt matching strategy

        Args:
            gt (torch.Tensor): (B, G, 4) -> (B, 1, G, 4)
            df (torch.Tensor): (P, 4) -> (1, P, 1, 4)
            threshold (float, optional): threshold of iou. Defaults to 0.5.

        Returns:
            torch.Tensor: matching mask
        """
        gt = gt.unsqueeze(1)
        df = df.unsqueeze(0).unsqueeze(2)

        g_cx, g_cy, g_w, g_h = [gt[:, :, :, i] for i in range(4)]
        d_cx, d_cy, d_w, d_h = [df[:, :, :, i] for i in range(4)]
        w = (torch.min(g_cx + g_w/2, d_cx + d_w/2) - torch.max(g_cx - g_w/2, d_cx - d_w/2)).clamp(min=0)
        h = (torch.min(g_cy + g_h/2, d_cy + d_h/2) - torch.max(g_cy - g_h/2, d_cy - d_h/2)).clamp(min=0)

        return (w * h / (g_w * g_h + d_w * d_h - w * h)) > threshold

    def calc_delta(self, gt: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
        """calculate g-hat

        Args:
            gt (torch.Tensor): (B, G, 4) -> (B, 1, G, 4)
            df (torch.Tensor): (1, P, 1, 4)

        Returns:
            torch.Tensor: g-hat tensor
        """
        gt = gt.unsqueeze(1)
        df = df.unsqueeze(0).unsqueeze(2)

        g_cx, g_cy, g_w, g_h = [gt[:, :, :, i] for i in range(4)]
        d_cx, d_cy, d_w, d_h = [df[:, :, :, i] for i in range(4)]
        g_cx = (g_cx - d_cx) / d_w
        g_cy = (g_cy - d_cy) / d_h
        g_w = torch.where(g_w > 0, torch.log(g_w / d_w), g_w)
        g_h = torch.where(g_w > 0, torch.log(g_w / d_w), g_w)

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
            pr (torch.Tensor): (B, P, class_num) -> (B, P, 1, class_num)
            gt (torch.Tensor): (B, G, class_num) -> (B, 1, G, class_num)

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
            pos_num (torch.Tensor): (B)
            neg_num (torch.Tensor): (B)

        Returns:
            torch.Tensor: (B)
        """
        cond = pos_num * 3 > neg_num
        return torch.where(cond, neg_num // 3, pos_num), torch.where(cond, neg_num, pos_num * 3)
    
    def kthvalue(self, tensor: torch.Tensor, k: torch.Tensor, mode: str) -> torch.Tensor:
        """get kthvalue from tensor

        Args:
            tensor (torch.Tensor): (P)
            k (torch.Tensor): (1)
            mode (str): 'min' or 'max'

        Returns:
            torch.Tensor: kth value
        """        
        if k > 0:
            return torch.kthvalue(tensor, k=k).values
        else:
            if mode == 'min':
                return torch.kthvalue(tensor, k=1).values - 1
            elif mode == 'max':
                return torch.kthvalue(tensor, k=1).values + 1
