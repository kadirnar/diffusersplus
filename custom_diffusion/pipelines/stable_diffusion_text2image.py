from typing import List, Optional

import torch
from diffusers import StableDiffusionPipeline

from custom_diffusion.utils.scheduler_utils import get_scheduler


class StableDiffusionText2ImgGenerator:
    """
    A class to handle image generation using stable diffusion and control net models.
    """

    def __init__(self):
        self.pipe = None
        self.model_cache = {}

    def _load_stable_diffusion_pipeline(self, stable_model_path: str = "runwayml/stable-diffusion-v1-5"):
        """
        This function loads the stable diffusion pipeline.

        Args:
        stable_model_path (str): Path to the stable diffusion pipeline.

        """
        self.pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=stable_model_path,
            safety_checker=None,
            torch_dtype=torch.float16,
        )

    def load_model(
        self,
        stable_model_path: str = "runwayml/stable-diffusion-v1-5",
        scheduler_name: str = "DDIM",
    ):
        """
        Load the models and setup the scheduler if not cached.

        Args:
        stable_model_path (str): Path to the stable model.
        controlnet_model_path (str): Path to the controlnet model.
        scheduler_name (str): Name of the scheduler.

        Returns:
        pipe: Configured model pipeline.
        """
        model_key = (stable_model_path, scheduler_name)

        # load and setup models only if they're not in the cache
        if model_key not in self.model_cache:
            self._load_stable_diffusion_pipeline(stable_model_path)
            self.pipe = get_scheduler(pipe=self.pipe, scheduler_name=scheduler_name)
            self.pipe.to("cuda")
            self.pipe.enable_xformers_memory_efficient_attention()

            self.model_cache[model_key] = self.pipe

        return self.model_cache[model_key]

    def _setup_generator(self, generator_seed: int = 0):
        if generator_seed == 0:
            random_seed = torch.randint(0, 1000000, (1,))
            generator = torch.manual_seed(random_seed)
        else:
            generator = torch.manual_seed(generator_seed)
        return generator

    def generate_image(
        self,
        stable_model_path: str = "runwayml/stable-diffusion-v1-5",
        controlnet_model_path: str = "lllyasviel/control_v11p_sd15_canny",
        scheduler_name: str = "DDIM",
        prompt: List[str] = ["A photo of a cat."],
        negative_prompt: List[str] = ["bad"],
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 20,
        guidance_scale: int = 7.0,
        guidance_rescale: float = 0.0,
        generator_seed: int = 0,
        height: int = 512,
        width: int = 512,
    ):
        """
        This function generates an image based on the given parameters.

        Args:
        stable_model_path (str): Path to the stable model.
        controlnet_model_path (str): Path to the controlnet model.
        scheduler_name (str): Name of the scheduler.
        image_path (dict): Path of the images.
        prompt (dict): The prompt for image generation.
        negative_prompt (str): The negative prompt for image generation.
        height (int): The height of the image to generate.
        width (int): The width of the image to generate.
        guess_mode (bool): Whether or not to use guess mode in image generation.
        num_images_per_prompt (int): The number of images to generate per prompt.
        num_inference_steps (int): The number of inference steps in image generation.
        guidance_scale (int): The scale of guidance in image generation.
        controlnet_conditioning_scale (int): The scale of controlnet conditioning in image generation.
        generator_seed (int): The seed for the random generator.
        preprocess_type (str): The type of preprocessing to apply.

        Returns:
        output: The generated image.
        """
        pipe = self.load_model(
            stable_model_path=stable_model_path,
            controlnet_model_path=controlnet_model_path,
            scheduler_name=scheduler_name,
        )

        generator = self._setup_generator(generator_seed)

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            height=height,
            width=width,
            generator=generator,
        ).images

        return output
