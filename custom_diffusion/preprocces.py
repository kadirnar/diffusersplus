import cv2
import numpy as np
from controlnet_aux import HEDdetector, MLSDdetector, OpenposeDetector
from PIL import Image


def depth_image(image):
    """
    Generate a depth image from a 2D input image using a depth-estimation model.

    Args:
    image (PIL.Image.Image): Input 2D image.

    Returns:
    PIL.Image.Image: Depth image.
    """
    from transformers import pipeline

    depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-hybrid-midas")
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)

    return control_image


def canny_image(image):
    """
    Apply Canny edge detection to a 2D input image.

    Args:
    image (PIL.Image.Image): Input 2D image.

    Returns:
    PIL.Image.Image: Image after applying Canny edge detection.
    """
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)

    return control_image


def hed_image(image):
    """
    Generate a HED edge detection image from a 2D input image.

    Args:
    image (str): File path to the input 2D image.

    Returns:
    torch.Tensor: Image after applying HED edge detection.
    """
    hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")
    image = hed(image)

    return image


def mlsd_image(image):
    """
    Generate a MLSD line detection image from a 2D input image.

    Args:
    image (str): File path to the input 2D image.

    Returns:
    torch.Tensor: Image after applying MLSD line detection.
    """

    mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
    image = mlsd(image)

    return image


def normal_image(image):
    """
    Generate a normalized depth image from a 2D input image.

    Args:
    image (str): File path to the input 2D image.

    Returns:
    PIL.Image.Image: Normalized depth image.
    """

    from transformers import pipeline

    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
    image = depth_estimator(image)["predicted_depth"][0]
    image = image.numpy()
    image_depth = image.copy()
    image_depth -= np.min(image_depth)
    image_depth /= np.max(image_depth)
    bg_threhold = 0.4
    x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    x[image_depth < bg_threhold] = 0
    y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    y[image_depth < bg_threhold] = 0
    z = np.ones_like(x) * np.pi * 2.0
    image = np.stack([x, y, z], axis=2)
    image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
    image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)

    return image


def pose_image(image):
    """
    Generate an OpenPose human pose estimation image from a 2D input image.

    Args:
    image (str): File path to the input 2D image.

    Returns:
    torch.Tensor: Image after applying OpenPose human pose estimation.
    """

    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    image = openpose(image)

    return image


def scribble_image(image):
    """
    Generate a scribbled image from a 2D input image using HED edge detection.

    Args:
    image (str): File path to the input 2D image.

    Returns:
    torch.Tensor: Scribbled image after applying HED edge detection.
    """
    hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")
    image = hed(image, scribble=True)

    return image


def ade_palette():
    """
    Define and return the ADE20K color palette used for semantic segmentation.

    Each color in the palette is assigned to a specific class,
    allowing you to color code the output of a semantic segmentation algorithm.

    Returns:
    List[List[int]]: A list of RGB color values. Each color is itself a list of three integers.
    """
    return [
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ]


def segmentation_image(image):
    """
    Perform semantic segmentation on an image using a pre-trained UperNet model.

    Args:
    image_path (str): File path to the input image.

    Returns:
    PIL.Image.Image: Image with the regions segmented and color-coded according to the ADE20K palette.
    """
    import torch
    from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

    image = image.convert("RGB")
    pixel_values = image_processor(image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = image_segmentor(pixel_values)

    seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    palette = np.array(ade_palette())

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg.astype(np.uint8)
    image = Image.fromarray(color_seg)

    return image


preprocces_dicts = {
    "Depth-Midas": depth_image,
    "Canny": canny_image,
    "Hed": hed_image,
    "Mlsd": mlsd_image,
    "Normal": normal_image,
    "Pose": pose_image,
    "Scribble": scribble_image,
    "Segmentation": segmentation_image,
}
