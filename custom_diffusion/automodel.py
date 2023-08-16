# Mapping between task identifiers and their respective class names
TASK_ID_TO_CLASS_MAPPING = {
    "stable-txt2img": "StableDiffusionText2ImgGenerator",
    "stable-img2img": "StableDiffusionImg2ImgGenerator",
    "stable-inpaint": "StableDiffusionInpaintingGenerator",
    "stable-upscale": "StableDiffusionUpscaleGenerator",
    "stable-pix2pix": "StableDiffusionInstructPix2PixGenerator",
    "controlnet": "StableDiffusionControlNetGenerator",
    "controlnet-img2img": "StableDiffusionControlNetImg2ImgGenerator",
    "controlnet-inpaint": "StableDiffusionControlNetInpaintGenerator",
}


class AutoDiffusionModel:
    """
    Factory class for automatically generating diffusion models based on task IDs.
    Provides a convenient interface to load models without knowing the underlying classes.
    """

    @staticmethod
    def from_pretrained(
        task_id: str = "stable-txt2img",
        model_path: str = "runwayml/stable-diffusion-v1-5",
        scheduler_name: str = "DDIM",
        **kwargs,
    ):
        """
        Load a diffusion model based on its task ID, model path, scheduler name, and other parameters.

        Args:
        - task_id (str, optional): Identifier for the type of diffusion task. Defaults to 'stable-txt2img'.
        - model_path (str, optional): Path to the pretrained weights. If not provided, model is initialized with default path.
        - scheduler_name (str, optional): Name of the scheduler. Defaults to 'DDIM'.
        - **kwargs: Additional arguments required by the specific diffusion model.

        Returns:
        - model_instance (BaseDiffusionModel): Initialized diffusion model.
        """

        # Get the class name based on task ID
        model_class_name = TASK_ID_TO_CLASS_MAPPING.get(task_id)
        if not model_class_name:
            raise ValueError(f"Unsupported task ID: {task_id}")

        # Dynamically import the required class using its name and the task ID
        DiffusionModelClass = getattr(
            __import__(f"custom_diffusion.pipelines.{task_id}", fromlist=[model_class_name]), model_class_name
        )

        # Return an instance of the loaded model class, initialized with the provided parameters
        model_instance = DiffusionModelClass()
        model_instance.load_model(model_path=model_path, scheduler_name=scheduler_name, **kwargs)

        return model_instance
