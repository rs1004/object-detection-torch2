import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self, weights_path=None, num_classes=20, transfer_learning=False):
        # initialize
        super(VGG16, self).__init__()
        self.transfer_learning = transfer_learning
        self.mean = torch.Tensor([0.485, 0.456, 0.406])
        self.std = torch.Tensor([0.229, 0.224, 0.225])

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
        x = self.normalize(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self.transfer_learning:
            x = self.classifier2(x)
        else:
            x = self.classifier(x)
        return x

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """normalize tensor

        Args:
            x (torch.Tensor): tensor

        Returns:
            torch.Tensor: normalized tensor
        """
        mean = self.mean.reshape(1, 3, 1, 1).to(x.device)
        std = self.std.reshape(1, 3, 1, 1).to(x.device)
        x = x.sub(mean).div(std)
        return x

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """calculate loss

        Args:
            outputs (torch.Tensor): (N, C)
            targets (torch.Tensor): (N, C)

        Returns:
            torch.Tensor: loss
        """
        outputs = F.log_softmax(outputs, dim=1)
        loss = torch.sum(targets * outputs, dim=1).mean()
        return loss
