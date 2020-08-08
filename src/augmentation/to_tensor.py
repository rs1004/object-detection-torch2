from torchvision.transforms import ToTensor


class ToTensor(ToTensor):
    """This is an extension of torchvision.transforms.ToTensor so that it can be applied to 'image' and 'gt'.
    """

    def __call__(self, img, gt):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return super(ToTensor, self).__call__(img), gt
