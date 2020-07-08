import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self, class_num, is_train=True):
        # initialize
        super(VGG16, self).__init__()
        self.is_train = is_train

        # layer1
        self.conv1_1 = ConvWithBN(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = ConvWithBN(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer2
        self.conv2_1 = ConvWithBN(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = ConvWithBN(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer3
        self.conv3_1 = ConvWithBN(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = ConvWithBN(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = ConvWithBN(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer4
        self.conv4_1 = ConvWithBN(in_channels=256, out_channels=512, kernel_size=3, padding=2)
        self.conv4_2 = ConvWithBN(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = ConvWithBN(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer5
        self.conv5_1 = ConvWithBN(in_channels=512, out_channels=512, kernel_size=3, padding=2)
        self.conv5_2 = ConvWithBN(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = ConvWithBN(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer6
        self.fc6_1 = nn.Linear(in_features=512 * 10 * 10, out_features=4096)
        self.fc6_2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc6_3 = nn.Linear(in_features=4096, out_features=class_num)

    def forward(self, x):
        # layer1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        # layer2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        # layer3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        # layer4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)

        # layer5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)

        # layer6
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc6_1(x))
        if self.is_train:
            x = nn.Dropout(p=0.2)(x)
        x = F.relu(self.fc6_2(x))
        if self.is_train:
            x = nn.Dropout(p=0.2)(x)
        x = self.fc6_3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def loss(self, output, target):
        output = F.softmax(output, dim=1)
        loss = nn.CrossEntropyLoss()(input=output, target=target)
        return loss


class ConvWithBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvWithBN, self).__init__()
        self.conv = ConvWithBN(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = F.relu(x)
        return x
