import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self, weights_path=None, num_classes=20, transfer_learning=False):
        # initialize
        super(VGG16, self).__init__()
        self.transfer_learning = transfer_learning

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

        # classification layer2 (for transfer learning)
        self.classifier2 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        # load weights
        if weights_path and weights_path.exists():
            print('weights loaded.')
            self.load_state_dict(torch.load(weights_path.as_posix()))
        else:
            vgg16_bn = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16_bn', pretrained=True)
            self.load_state_dict(vgg16_bn.state_dict(), strict=False)
            self._initialize_weights()

        if self.transfer_learning:
            for m in self.features.modules():
                for param in m.parameters():
                    param.requires_grad = False

    def _initialize_weights(self):
        for m in self.classifier2.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self.transfer_learning:
            x = self.classifier2(x)
        else:
            x = self.classifier(x)
        return x

    def loss(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        loss = nn.CrossEntropyLoss()(input=outputs, target=targets)
        return loss


class SSD(nn.Module):
    def __init__(self, num_classes, weights_path=None, weights_path_vgg16=None):
        # initialize
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.default_bboxes = self._get_default_bboxes()

        features = nn.ModuleDict()

        # vgg16 layer
        vgg16 = VGG16(weights_path=weights_path_vgg16)
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
            'act_4_3': nn.Conv2d(in_channels=512, out_channels=4 * (num_classes + 4), kernel_size=3, padding=1),
            'act_7_1': nn.Conv2d(in_channels=1024, out_channels=6 * (num_classes + 4), kernel_size=3, padding=1),
            'act_8_2': nn.Conv2d(in_channels=512, out_channels=6 * (num_classes + 4), kernel_size=3, padding=1),
            'act_9_2': nn.Conv2d(in_channels=256, out_channels=6 * (num_classes + 4), kernel_size=3, padding=1),
            'act_10_2': nn.Conv2d(in_channels=256, out_channels=4 * (num_classes + 4), kernel_size=3, padding=1),
            'act_11_2': nn.Conv2d(in_channels=256, out_channels=4 * (num_classes + 4), kernel_size=3, padding=1),
        })

        # load weights
        if weights_path and weights_path.exists():
            print('weights loaded.')
            self.load_state_dict(torch.load(weights_path.as_posix()))
        else:
            self._initialize_weights()

    def forward(self, x):
        batch_size = x.size(0)
        y = torch.empty((batch_size, 0, self.num_classes + 4))
        y = y.to(x.device)

        for name, layer in self.features.items():
            x = layer(x)
            if name in self.classifier:
                y_ = self.classifier[name](x).permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes + 4)
                y = torch.cat([y, y_], dim=1)

        return y

    def _get_default_bboxes(self):
        def s_(k, m=6, s_min=0.2, s_max=0.9):
            return s_min + (s_max - s_min) * (k - 1) / (m - 1)

        default_bboxes = torch.empty(0, 4)
        cfg = [[38, 38, 4], [19, 19, 6], [10, 10, 6], [5, 5, 6], [3, 3, 4], [1, 1, 4]]

        for k, (m, n, a_num) in enumerate(cfg, start=1):
            aspects = [1, 2, 1 / 2, 'add'] if a_num == 4 else [1, 2, 1 / 2, 3, 1 / 3, 'add']
            for i in range(m):
                for j in range(n):
                    for a in aspects:
                        if a == 'add':
                            w = h = (s_(k) * s_(k + 1)) ** 0.5
                        else:
                            w = s_(k) * (a ** 0.5)
                            h = s_(k) * ((1 / a) ** 0.5)
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

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor, default_bboxes: torch.Tensor, a: int = 1) -> torch.Tensor:
        """calculate loss

        Args:
            outputs (torch.Tensor)   : (N, P, 4 + C)
            targets (torch.Tensor)     : (N, G, 4 + C)
            default_bboxes (torch.Tensor): (P, 4)
            a (int, optional): weight term of loss formula. Defaults to 1.

        Returns:
            torch.Tensor: loss
        """
        # constant definition
        N = outputs.shape[0]
        P = outputs.shape[1]
        C = outputs.shape[2] - 4

        # matching
        is_match = self.match(gt=targets, df=default_bboxes)

        # localization loss
        l = outputs[:, :, :4].unsqueeze(2)
        g = self.calc_delta(gt=targets, df=default_bboxes)
        l_loc = (self.smooth_l1(l - g).sum(dim=3) * is_match).sum(dim=2)

        # confidence loss
        # positive
        softmax_pos = self.softmax_cross_entropy(pr=outputs[:, :, 4:], gt=targets[:, :, 4:])
        l_conf_pos = (softmax_pos * is_match).sum(dim=2)

        # negative
        gt_void = torch.eye(C)[0].unsqueeze(0).unsqueeze(1).to(outputs.device)
        softmax_neg = self.softmax_cross_entropy(pr=outputs[:, :, 4:], gt=gt_void)
        is_not_match = is_match.sum(dim=2, keepdims=True) == 0
        l_conf_neg = (softmax_neg * is_not_match).sum(dim=2)

        # hard negative mining
        pos_num = (is_match.sum(dim=2) != 0).sum(dim=1)
        neg_num = P - pos_num
        pos_num, neg_num = self.split_pos_neg(pos_num, neg_num)

        pos_valid = l_conf_pos > torch.stack([self.k_plus_1_th_value(l_conf_pos[i], pos_num[i]) for i in range(N)]).unsqueeze(1)
        neg_valid = l_conf_neg > torch.stack([self.k_plus_1_th_value(l_conf_neg[i], neg_num[i]) for i in range(N)]).unsqueeze(1)

        # calculate loss (if pos_num = 0, then loss = 0)
        pos_num = torch.where(pos_num > 0, 1 / pos_num.float(), pos_num.float())
        loss = (((a * l_loc + l_conf_pos) * pos_valid + l_conf_neg * neg_valid).sum(dim=1) * pos_num).mean()

        return loss

    def match(self, gt: torch.Tensor, df: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """adapt matching strategy

        Args:
            gt (torch.Tensor): (N, G, 4) -> (N, 1, G, 4)
            df (torch.Tensor): (P, 4) -> (1, P, 1, 4)
            threshold (float, optional): threshold of iou. Defaults to 0.5.

        Returns:
            torch.Tensor (N, P, G): matching mask
        """
        gt = gt.unsqueeze(1)
        df = df.unsqueeze(0).unsqueeze(2)

        g_cx, g_cy, g_w, g_h = [gt[:, :, :, i] for i in range(4)]
        d_cx, d_cy, d_w, d_h = [df[:, :, :, i] for i in range(4)]
        w = (torch.min(g_cx + g_w / 2, d_cx + d_w / 2) - torch.max(g_cx - g_w / 2, d_cx - d_w / 2)).clamp(min=0)
        h = (torch.min(g_cy + g_h / 2, d_cy + d_h / 2) - torch.max(g_cy - g_h / 2, d_cy - d_h / 2)).clamp(min=0)

        return torch.where(g_w * g_h > 0, w * h / (g_w * g_h + d_w * d_h - w * h), g_w * g_h) > threshold

    def calc_delta(self, gt: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
        """calculate g-hat

        Args:
            gt (torch.Tensor): (N, G, 4) -> (N, 1, G, 4)
            df (torch.Tensor): (1, P, 1, 4)

        Returns:
            torch.Tensor (N, P, G, 4): g-hat tensor
        """
        gt = gt.unsqueeze(1)
        df = df.unsqueeze(0).unsqueeze(2)

        g_cx, g_cy, g_w, g_h = [gt[:, :, :, i] for i in range(4)]
        d_cx, d_cy, d_w, d_h = [df[:, :, :, i] for i in range(4)]
        g_cx = (g_cx - d_cx) / d_w
        g_cy = (g_cy - d_cy) / d_h
        g_w = torch.where(g_w > 0, torch.log(g_w / d_w), g_w)
        g_h = torch.where(g_h > 0, torch.log(g_h / d_h), g_h)

        return torch.stack([g_cx, g_cy, g_w, g_h], dim=3)

    def smooth_l1(self, x: torch.Tensor) -> torch.Tensor:
        """calculate smooth l1

        Args:
            x (torch.Tensor): any tensor

        Returns:
            torch.Tensor (N, P, G, 4): smooth l1
        """
        return torch.where(x.abs() < 1, 0.5 * x * x, x.abs() - 0.5)

    def softmax_cross_entropy(self, pr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """calculate softmax cross-entropy

        Args:
            pr (torch.Tensor): (N, P, num_classes) -> (N, P, 1, num_classes)
            gt (torch.Tensor): (N, G, num_classes) -> (N, 1, G, num_classes)

        Returns:
            torch.Tensor (N, P, G): softmax cross-entropy
        """
        pr = pr.unsqueeze(2)
        gt = gt.unsqueeze(1)

        return -(gt * F.log_softmax(pr, dim=3)).sum(dim=3)

    def split_pos_neg(self, pos_num: torch.Tensor, neg_num: torch.Tensor) -> tuple:
        """split pos:neg = 1:3

        Args:
            pos_num (torch.Tensor): (N)
            neg_num (torch.Tensor): (N)

        Returns:
            tuple: (pos_num, neg_num)
        """
        cond = pos_num * 3 > neg_num
        return torch.where(cond, neg_num // 3, pos_num), torch.where(cond, neg_num, pos_num * 3)

    def k_plus_1_th_value(self, tensor: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """get (k+1)-th largest value from tensor

        Args:
            tensor (torch.Tensor): (P)
            k (torch.Tensor): (1)

        Returns:
            torch.Tensor (1):
                * k > 0: (k+1)-th largest value
                * k = 0: max value
        """
        if k > 0:
            return torch.kthvalue(tensor, k=len(tensor) - k).values
        else:
            return tensor.max()
