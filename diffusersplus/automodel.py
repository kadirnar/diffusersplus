from typing import Optional

from diffusersplus.pipelines import (
    StableDiffusionControlNetGenerator,
    StableDiffusionControlNetImg2ImgGenerator,
    StableDiffusionControlNetInpaintGenerator,
    StableDiffusionImg2ImgGenerator,
    StableDiffusionT2iAdapterGenerator,
    StableDiffusionText2ImgGenerator,
    StableDiffusionUpscaleGenerator,
    StableDiffusionXLImageGenerator,
)
from diffusersplus.pipelines.base import BaseDiffusionModel

TASK_ID_TO_CLASS_MAPPING = {
    "stable-txt2img": StableDiffusionText2ImgGenerator,
    "stable-img2img": StableDiffusionImg2ImgGenerator,
    "stable-upscale": StableDiffusionUpscaleGenerator,
    "controlnet": StableDiffusionControlNetGenerator,
    "controlnet-img2img": StableDiffusionControlNetImg2ImgGenerator,
    "controlnet-inpaint": StableDiffusionControlNetInpaintGenerator,
    "controlnet-sdxl": StableDiffusionXLImageGenerator,
    "controlnet-t2i_adapter": StableDiffusionT2iAdapterGenerator,
}


def diffusion_pipeline(
    task_id: str,
    stable_model_id: str = "runwayml/stable-diffusion-v1-5",
    controlnet_model_id: Optional[str] = "lllyasviel/control_v11p_sd15_canny",
    vae_model_id: Optional[str] = "madebyollin/sdxl-vae-fp16-fix",
    adapter_model_id: Optional[str] = "TencentARC/t2iadapter_canny_sd15v2",
    scheduler_name: str = "DDIM",
) -> BaseDiffusionModel:
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

    model_instance = DiffusionModelClass(
        stable_model_id=stable_model_id,
        controlnet_model_id=controlnet_model_id,
        vae_model_id=vae_model_id,
        adapter_model_id=adapter_model_id,
        scheduler_name=scheduler_name,
    )

    return model_instance
