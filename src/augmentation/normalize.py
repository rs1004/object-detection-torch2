from torchvision.transforms import Normalize


class Normalize(Normalize):
    """This is an extension of torchvision.transforms.Normalize so that it can be applied to 'image' and 'gt'.
    """

    def __init__(self, mean, std, inplace=False):
        super(Normalize, self).__init__(mean, std, inplace)

    def __call__(self, image, gt):
        return super(Normalize, self).__call__(image), gt
