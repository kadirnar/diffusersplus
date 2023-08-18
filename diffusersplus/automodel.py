from diffusersplus.pipelines import (
    StableDiffusionControlNetGenerator,
    StableDiffusionControlNetImg2ImgGenerator,
    StableDiffusionControlNetInpaintGenerator,
    StableDiffusionImg2ImgGenerator,
    StableDiffusionText2ImgGenerator,
    StableDiffusionUpscaleGenerator,
)

from diffusersplus.pipelines.base import BaseDiffusionModel

from typing import List, Optional


TASK_ID_TO_CLASS_MAPPING = {
    "stable-txt2img": StableDiffusionText2ImgGenerator,
    "stable-img2img": StableDiffusionImg2ImgGenerator,
    "stable-upscale": StableDiffusionUpscaleGenerator,
    "controlnet": StableDiffusionControlNetGenerator,
    "controlnet-img2img": StableDiffusionControlNetImg2ImgGenerator,
    "controlnet-inpaint": StableDiffusionControlNetInpaintGenerator,
}


def diffusion_pipeline(task_id: str, stable_model_id: str = None, controlnet_model_id: str = None, scheduler_name: str = None) -> BaseDiffusionModel:
    """
    Create and return an instance of the specified diffusion model based on the task ID.

    Args:
    - task_id (str): The task identifier.
    - stable_model_id (str, optional): Identifier for the stable model.
    - controlnet_model_id (str, optional): Identifier for the control net model.
    - scheduler_name (str, optional): Name of the scheduler. Default is "DDIM".

    Returns:
    - BaseDiffusionModel: An instance of the specified diffusion model.
    """
    DiffusionModelClass = TASK_ID_TO_CLASS_MAPPING.get(task_id)
    if not DiffusionModelClass:
        raise ValueError(f"Unsupported task ID: {task_id}")

    model_instance = DiffusionModelClass(stable_model_id=stable_model_id, controlnet_model_id=controlnet_model_id, scheduler_name=scheduler_name)

    return model_instance
