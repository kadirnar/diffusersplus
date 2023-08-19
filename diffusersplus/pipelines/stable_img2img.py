from typing import Optional

import torch
from diffusers import StableDiffusionImg2ImgPipeline

from diffusersplus.pipelines.base import BaseDiffusionModel
from diffusersplus.utils.data_utils import load_and_resize_image


class StableDiffusionImg2ImgGenerator(BaseDiffusionModel):
    """
    A class to handle image generation using stable diffusion models for image-to-image tasks.
    Inherits from the BaseDiffusionModel to utilize core functionalities.
    """

    def __init__(self, stable_model_id: str = None, controlnet_model_id: str = None, scheduler_name: str = None):
        super().__init__()
        self.stable_model_id = stable_model_id
        self.controlnet_model_id = controlnet_model_id
        self.scheduler_name = scheduler_name

    def _load_diffusion_pipeline(self):
        """
        Load the stable diffusion pipeline specific to image-to-image generation.
        """
        if not hasattr(self, "pipe") or self.pipe is None:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                pretrained_model_name_or_path=self.stable_model_id,
                safety_checker=None,
                torch_dtype=torch.float16,
            )
            self.load_scheduler("stable", self.stable_model_id, self.scheduler_name)

    def __call__(
        self,
        image_path: str,
        prompt: str = "A photo of a cat.",
        negative_prompt: str = "bad",
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: int = 7.0,
        strength: float = 0.5,
        generator_seed: int = 0,
        resize_type: str = "center_crop_and_resize",
        crop_size: Optional[int] = 512,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
    ) -> torch.Tensor:
        """
        Generate an image based on the provided parameters.
        """

        # Modeli yükle
        self._load_diffusion_pipeline()

        read_image = load_and_resize_image(
            image_path=image_path, resize_type=resize_type, height=height, width=width, crop_size=crop_size
        )

        generator = self._configure_random_generator(generator_seed)

        output = self.pipe(
            prompt=prompt,
            image=read_image,
            strength=strength,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images

        return output
