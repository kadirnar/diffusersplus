from typing import List, Optional

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from PIL import Image

from custom_diffusion.pipelines.base import BaseDiffusionModel
from custom_diffusion.preprocces import preprocces_dicts
from custom_diffusion.utils.data_utils import load_and_resize_image


class StableDiffusionControlNetImg2ImgGenerator(BaseDiffusionModel):
    """
    A class to handle image-to-image generation using stable diffusion and control net models.

    This class provides functionalities to generate images using a combination of stable diffusion
    and control net models. The models can be specified using their paths, and other parameters
    can be adjusted to fine-tune the image generation process.

    Example:
        ```
        generator = StableDiffusionControlNetImg2ImgGenerator()
        generated_image = generator.generate_output(
            image_path="path_to_image.png",
            model_path="model_path",
            controlnet_model_path="controlnet_model_path",
            scheduler_name="DDIM",
            prompt="A clear high-resolution image of a cat.",
        )
        ```

    """

    def _load_diffusion_pipeline(
        self,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        controlnet_model_path: str = "lllyasviel/control_v11p_sd15_canny",
    ):
        """
        Load the stable diffusion pipeline with control net model.

        Args:
            model_path (str): Path to the stable diffusion pipeline with control net model.
            controlnet_model_path (str): Path to the control net model.
        """
        controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16,
        )

    def generate_output(
        self,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        controlnet_model_path: str = "lllyasviel/sd-controlnet-canny",
        scheduler_name: str = "DDIM",
        image_path: str = "test.png",
        prompt: str = "A photo of a cat.",
        negative_prompt: str = "bad",
        height: int = 512,
        width: int = 512,
        preprocess_type: str = "Canny",
        resize_type: str = "center_crop_and_resize",
        guess_mode: bool = False,
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 20,
        guidance_scale: int = 7.0,
        controlnet_conditioning_scale: int = 1.0,
        strength: float = 0.5,
        generator_seed: int = 0,
    ) -> torch.Tensor:
        """
        Generate an image based on the provided parameters.

        Args:
            ... [Similar to the previous version but adjusted to single image and prompt strings]

        Returns:
            output (torch.Tensor): The generated image.
        """

        # Load image and preprocess
        read_image = load_and_resize_image(image_path=image_path, resize_type=resize_type, height=height, width=width)
        control_image = preprocces_dicts[preprocess_type](read_image)

        # Load model and set up pipeline
        pipe = self.load_model(model_path=model_path, scheduler_name=scheduler_name)
        self._load_diffusion_pipeline(model_path=model_path, controlnet_model_path=controlnet_model_path)
        generator = self._setup_generator(generator_seed)

        # Generate the image
        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            guess_mode=guess_mode,
            control_image=control_image,
            image=read_image,
            strength=strength,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            generator=generator,
        ).images

        return output
