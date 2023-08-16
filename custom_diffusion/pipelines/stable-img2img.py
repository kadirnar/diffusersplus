from typing import Optional

import torch
from diffusers import StableDiffusionImg2ImgPipeline

from custom_diffusion.pipelines.base import BaseDiffusionModel
from custom_diffusion.utils.data_utils import load_and_resize_image


class StableDiffusionImg2ImgGenerator(BaseDiffusionModel):
    """
    A class to handle image generation using stable diffusion models for image-to-image tasks.
    Inherits from the BaseDiffusionModel to utilize core functionalities.

    Example:
    ```python
    generator = StableDiffusionImg2ImgGenerator()
    output = generator.generate_output(
        model_path="runwayml/stable-diffusion-v1-5",
        scheduler_name="DDIM",
        images_path_list=["input_image.jpg"]
    )
    ```
    """

    def _load_diffusion_pipeline(self, model_path: str):
        """
        Load the stable diffusion pipeline specific to image-to-image generation.

        Args:
        - model_path (str): Path to the stable diffusion pipeline.
        """
        # Check if the model is already cached
        if model_path in self.model_cache:
            self.pipe = self.model_cache[model_path]
            return

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            safety_checker=None,
            torch_dtype=torch.float16,
        )
        self.model_cache[model_path] = self.pipe

    def generate_output(
        self,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        scheduler_name: str = "DDIM",
        image_path: str = "test.png",
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

        Args:
        - model_path (str): Path to the stable model.
        - scheduler_name (str): Name of the scheduler.
        - image_path (str): Path to the input image.
        - prompt (str): The prompt for image guidance.
        - negative_prompt (str): The negative prompt for image guidance.
        - height (int): The height of the output image.
        - width (int): The width of the output image.
        - num_images_per_prompt (int): The number of images to generate per prompt.
        - num_inference_steps (int): The number of inference steps in image generation.
        - guidance_scale (int): The scale of guidance in image generation.
        - strength (float): The strength parameter for the diffusion process.
        - generator_seed (int): The seed for the random generator.
        - resize_type (str): The type of resizing to apply to the input image.
        - crop_size (int): The size of the crop if the "center_crop_and_resize" method is chosen.

        Returns:
        - output (torch.Tensor): The generated image.
        """

        read_image = load_and_resize_image(
            image_path=image_path, resize_type=resize_type, height=height, width=width, crop_size=crop_size
        )

        pipe = self.load_model(model_path=model_path, scheduler_name=scheduler_name)

        generator = self._setup_generator(generator_seed)

        output = pipe(
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
