import torch
from torchvision.transforms.functional import hflip
from torchvision.transforms import ColorJitter, RandomErasing


class RandomColorJitter(ColorJitter):
    def __init__(self, p: float = 0.5, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5):
        super(RandomColorJitter, self).__init__(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, img, gt):
        if torch.rand(1) < self.p:
            img = super(RandomColorJitter, self).__call__(img)
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
            gt[:, 0] = 1 - gt[:, 0]
        return img, gt


class RandomErasing(RandomErasing):
    def __init__(self, p=0.5, scale=(0.01, 0.04), ratio=(0.5, 2.0), max_iter=1):
        super(RandomErasing, self).__init__(p=p, scale=scale, ratio=ratio)
        self.max_iter = max_iter

    def __call__(self, img, gt):
        iter = torch.randint(low=1, high=self.max_iter + 1, size=(1,)).item()
        for _ in range(iter):
            img = super(RandomErasing, self).__call__(img)
        return img, gt
