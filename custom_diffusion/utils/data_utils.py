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
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def load_images_from_folder(folder):
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
            img = Image.open(os.path.join(folder, filename))
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
