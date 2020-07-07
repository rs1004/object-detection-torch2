import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, is_train=True):
        pass

    def forward(self, x):
        pass

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
