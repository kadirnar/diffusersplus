from custom_diffusion.pipelines import (
    StableDiffusionControlNetGenerator,
    StableDiffusionControlNetImg2ImgGenerator,
    StableDiffusionControlNetInpaintGenerator,
    StableDiffusionImg2ImgGenerator,
    StableDiffusionText2ImgGenerator,
    StableDiffusionUpscaleGenerator,
)

TASK_ID_TO_CLASS_MAPPING = {
    "stable-txt2img": StableDiffusionText2ImgGenerator,
    "stable-img2img": StableDiffusionImg2ImgGenerator,
    "stable-upscale": StableDiffusionUpscaleGenerator,
    "controlnet": StableDiffusionControlNetGenerator,
    "controlnet-img2img": StableDiffusionControlNetImg2ImgGenerator,
    "controlnet-inpaint": StableDiffusionControlNetInpaintGenerator,
}


def pipeline(task_id: str, model_id: str, scheduler_name: str = "DDIM"):
    # TASK_ID_TO_CLASS_MAPPING dictionary is assumed to be defined as before

    # Get the class directly from the mapping based on task_id
    DiffusionModelClass = TASK_ID_TO_CLASS_MAPPING.get(task_id)
    if not DiffusionModelClass:
        raise ValueError(f"Unsupported task ID: {task_id}")

    # Instantiate the model class
    model_instance = DiffusionModelClass()

    # Load the model based on model_id
    model_instance.load_model(model_path=model_id, scheduler_name=scheduler_name)

    return model_instance
