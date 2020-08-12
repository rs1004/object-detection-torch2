import torch
from torchvision.transforms.functional import hflip
from torchvision.transforms import ColorJitter, RandomPerspective, RandomErasing


class RandomColorJitter(ColorJitter):
    def __init__(self, p: float = 0.5, brightness=0, contrast=0, saturation=0, hue=0):
        super(RandomColorJitter, self).__init__(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, img, gt):
        if torch.rand(1) < self.p:
            img = super(RandomColorJitter, self).__call__(img)
        return img, gt


class RandomPerspective(RandomPerspective):
    def __init__(self, distortion_scale=0.5, p=0.5):
        super(RandomPerspective, self).__init__(distortion_scale=distortion_scale, p=p)

    def __call__(self, img, gt):
        img = super(RandomPerspective, self).__call__(img)
        return img, gt


class RandomFlip(torch.nn.Module):
    def __init__(self, p: float = 0.5):
        """initialize

        Args:
            p (float, optional): probability of executing. Defaults to 0.5.
        """
        self.p = p

    def __call__(self, img, gt):
        if torch.rand(1) < self.p:
            img = hflip(img)
            gt[:, 0:2] = 1 - gt[:, 0:2]
        return img, gt


class RandomScale(torch.nn.Module):
    def __init__(self, p: float = 0.5, ratio_range: tuple = (0.9, 1.1)):
        """initialize

        Args:
            p (float, optional): probability of executing. Defaults to 0.5.
            ratio_range (tuple, optional): (lower_limit, upper_limit). Defaults to (0.9, 1.1).
        """
        self.p = p
        self.ratio_range = ratio_range

    def __call__(self, img, gt):
        if torch.rand(1) < self.p:
            low, high = self.ratio_range
            ratio = low + torch.rand(1).item() * (high - low)
            gt[:, 2:4] = gt[:, 2:4] * ratio
        return img, gt


class RandomShift(torch.nn.Module):
    def __init__(self, p: float = 0.5, horizontal_upper_limit: float = 1.1, vertical_upper_limit: float = 1.1):
        """initialize

        Args:
            p (float, optional): probability of executing. Defaults to 0.5.
            horizontal_upper_limit (float, optional): upper limit of horizontal movement. Defaults to 1.1.
            vertical_upper_limit (float, optional): upper limit of vertical movement. Defaults to 1.1.
        """
        self.p = p
        self.vertical_upper_limit = vertical_upper_limit
        self.horizontal_upper_limit = horizontal_upper_limit

    def __call__(self, img, gt):
        if torch.rand(1) < self.p:
            sign = torch.randn(1) > 0
            horizontal_ratio = sign * (1. + torch.rand(1).item() * (self.horizontal_upper_limit - 1.))
            sign = torch.randn(1) > 0
            vertical_ratio = sign * (1. + torch.rand(1).item() * (self.vertical_upper_limit - 1.))

            gt[:, 0] = gt[:, 0] * horizontal_ratio
            gt[:, 1] = gt[:, 1] * vertical_ratio
        return img, gt


class RandomErasing(RandomErasing):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.5, 2.0), max_iter=1):
        super(RandomErasing, self).__init__(p=p, scale=scale, ratio=ratio)
        self.max_iter = max_iter

    def __call__(self, img, gt):
        iter = torch.randint(low=1, high=self.max_iter + 1, size=(1,)).item()
        for _ in range(iter):
            img = super(RandomErasing, self).__call__(img)
        return img, gt
