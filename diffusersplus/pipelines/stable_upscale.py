import torch
from diffusers import StableDiffusionUpscalePipeline

from diffusersplus.pipelines.base import BaseDiffusionModel
from diffusersplus.utils.data_utils import load_and_resize_image


class StableDiffusionUpscaleGenerator(BaseDiffusionModel):
    """
    A class for generating upscaled images using the stable diffusion model.
    """

    def __init__(
        self, 
        stable_model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_model_id: str = None,
        scheduler_name: str = "DDIM"
    ):
        
        super().__init__()
        self.stable_model_id = stable_model_id
        self.controlnet_model_id = controlnet_model_id
        self.scheduler_name = scheduler_name

    def _load_diffusion_pipeline(self):
        """
        Load the stable diffusion upscale pipeline.
        """
        if not hasattr(self, 'pipe') or self.pipe is None:
            self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
                pretrained_model_name_or_path=self.stable_model_id,
                safety_checker=None,
                torch_dtype=torch.float16,
                revision="fp16"
            )
            self.load_scheduler("stable", self.stable_model_id, self.scheduler_name)

    def __call__(
        self,
        image_path: str,
        prompt: str = "A photo of a cat.",
        negative_prompt: str = "bad",
        preprocess_type: str = "Canny",
        resize_type: str = "center_crop_and_resize",
        noise_level: int = 20,
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 20,
        guidance_scale: int = 7.0,
        generator_seed: int = 0,
    ) -> torch.Tensor:
        """
        Generate an upscaled image based on the provided parameters.
        """

        # Modeli yükle
        self._load_diffusion_pipeline()

        read_image = load_and_resize_image(image_path=image_path, resize_type=resize_type, height=512, width=512)

        generator = self._configure_random_generator(generator_seed)

        output = self.pipe(
            prompt=prompt,
            image=read_image,
            noise_level=noise_level,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images

        return output
