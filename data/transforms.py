def get_transform(new_size=None):
    """
    obtain the image transforms required for the input data
    :param new_size: size of the resized images
    :return: image_transform => transform object from TorchVision
    """
    from jittor.transform import ToTensor, ImageNormalize, Compose, Resize, RandomHorizontalFlip

    if new_size is not None:
        image_transform = Compose([
            RandomHorizontalFlip(),
            Resize(new_size),
            ToTensor(),
            ImageNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    else:
        image_transform = Compose([
            RandomHorizontalFlip(),
            ToTensor(),
            ImageNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    return image_transform
