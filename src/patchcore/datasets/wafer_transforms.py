from PIL import Image, ImageOps
from torchvision.transforms import functional as TF


DEFAULT_WAFER_BACKGROUND = (68, 1, 84)


def resize_pad(image, size, interpolation, fill=DEFAULT_WAFER_BACKGROUND):
    """Resize with aspect ratio preserved, then pad to a square."""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Cannot transform an empty image.")
    scale = min(size / width, size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image = TF.resize(image, [new_height, new_width], interpolation=interpolation)
    pad_left = (size - new_width) // 2
    pad_top = (size - new_height) // 2
    pad_right = size - new_width - pad_left
    pad_bottom = size - new_height - pad_top
    return ImageOps.expand(image, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)


def transform_wafer(image, resize, imagesize, interpolation, mode="resize_pad", fill=DEFAULT_WAFER_BACKGROUND):
    if mode == "resize_crop":
        image = TF.resize(image, resize, interpolation=interpolation)
        return TF.center_crop(image, [imagesize, imagesize])
    if mode == "resize_only":
        return TF.resize(image, [imagesize, imagesize], interpolation=interpolation)
    if mode == "resize_pad":
        return resize_pad(image, imagesize, interpolation, fill=fill)
    raise ValueError(f"Unknown wafer transform mode: {mode}")
