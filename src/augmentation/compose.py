from torchvision.transforms import Compose


class Compose(Compose):
    """This is an extension of torchvision.transforms.Compose so that it can be applied to 'image' and 'gt'.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        super(Compose, self).__init__(transforms)

    def __call__(self, img, gt):
        for t in self.transforms:
            img, gt = t(img, gt)
        return img, gt
