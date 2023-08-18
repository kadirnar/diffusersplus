from typing import Optional, Union

from PIL import Image


def image_grid(imgs, rows, cols):
    """
    This function takes a list of images and creates a grid of images from them.

    Args:
    imgs (list): List of images to be used in the grid.
    rows (int): Number of rows in the grid.

    Returns:
    grid (Image): The grid of images.
    """
    from PIL import Image

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def load_images_from_folder(folder, pil_image=True):
    """
    Loads all .jpg and .png images in a specified folder and returns them as a list of PIL.Image objects.

    Parameters:
    folder (str): The path to the folder containing the images.

    Returns:
    list: A list of PIL.Image objects.

    """
    import os

    from PIL import Image

    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            if pil_image:
                img = Image.open(os.path.join(folder, filename))
            else:
                img = os.path.join(folder, filename)
            if img is not None:
                images.append(img)
    return images


def center_crop_and_resize(image, crop_size, height, width):
    """
    Crops the center of an image based on the input crop size and then resizes the image.

    Parameters:
    image_path (str): The path to the image to be processed.
    crop_size (int): The size of the square crop. The function will crop a square with this side length from the center of the image.
    height (int): The height of the resized image.
    width (int): The width of the resized image.

    Returns:
    PIL.Image: The cropped and resized image.

    """
    # Calculate the center and the crop box coordinates
    width, height = image.size
    left = (width - crop_size) / 2
    upper = (height - crop_size) / 2
    right = (width + crop_size) / 2
    lower = (height + crop_size) / 2

    # Crop the image using the calculated coordinates
    cropped_img = image.crop((left, upper, right, lower))

    # Resize the cropped image to the specified size
    resized_img = cropped_img.resize((height, width))

    return resized_img


def load_and_resize_image(
    image_path: str = "test.png",
    resize_type: str = "center_crop_and_resize",
    crop_size: Optional[int] = 512,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
) -> Image.Image:
    """
    Load and resize the image based on the provided parameters.

    Args:
    - image_path (str): Path to the image.
    - resize_type (str): The type of resizing to apply.
    - crop_size (int): The size of the crop.
    - height (int): The height of the image after resizing.
    - width (int): The width of the image after resizing.

    Returns:
    - Image.Image: The resized and loaded PIL Image.
    """
    image = Image.open(image_path)

    if resize_type == "center_crop_and_resize":
        image = center_crop_and_resize(image, crop_size=crop_size, height=height, width=width)

    elif resize_type == "resize":
        image = image.resize((height, width))

    else:
        raise ValueError("Invalid resize type.")

    return image


def image_grid(imgs, rows, cols):
    """
    This function takes a list of images and creates a grid of images from them.

    Args:
    imgs (list): List of images to be used in the grid.
    rows (int): Number of rows in the grid.

    Returns:
    grid (Image): The grid of images.
    """

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
