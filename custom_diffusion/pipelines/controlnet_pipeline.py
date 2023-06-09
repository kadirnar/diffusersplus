from typing import List, Optional

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

from custom_diffusion.preprocces import preprocces_dicts
from custom_diffusion.utils.data_utils import center_crop_and_resize
from custom_diffusion.utils.scheduler_utils import get_scheduler


from typing import List, Optional

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline


class StableDiffusionControlNetGenerator:
    """
    A class to handle image generation using stable diffusion and control net models.
    """

    def __init__(self):
        self.controlnet = None
        self.pipe = None
        self.model_cache = {}

    def _load_controlnet_model(self, controlnet_model_path):
        """
        This function loads the controlnet model.

        Args:
        controlnet_model_path (str): Path to the controlnet model.

        """
        self.controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)

    def _load_stable_diffusion_pipeline(self, stable_model_path):
        """
        This function loads the stable diffusion pipeline.

        Args:
        stable_model_path (str): Path to the stable diffusion pipeline.

        """
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=stable_model_path,
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=torch.float16,
        )

    def load_model(self, stable_model_path, controlnet_model_path, scheduler_name):
        """
        Load the models and setup the scheduler if not cached.

        Args:
        stable_model_path (str): Path to the stable model.
        controlnet_model_path (str): Path to the controlnet model.
        scheduler_name (str): Name of the scheduler.

        Returns:
        pipe: Configured model pipeline.
        """
        model_key = (stable_model_path, controlnet_model_path, scheduler_name)

        # load and setup models only if they're not in the cache
        if model_key not in self.model_cache:
            self._load_controlnet_model(controlnet_model_path)
            self._load_stable_diffusion_pipeline(stable_model_path)
            self.pipe = get_scheduler(pipe=self.pipe, scheduler_name=scheduler_name)
            self.pipe.to("cuda")
            self.pipe.enable_xformers_memory_efficient_attention()

            self.model_cache[model_key] = self.pipe

        return self.model_cache[model_key]

    def load_and_resize_image(
        self, image_path: str, resize_type: str, crop_size: Optional[int], height: Optional[int], width: Optional[int]
    ):
        """
        This function loads and resizes the image.

        Args:
        image_path (str): Path to the image.
        resize_type (str): The type of resizing to apply.
        crop_size (int): The size of the crop.
        height (int): The height of the image to generate.
        width (int): The width of the image to generate.

        Returns:
        Image: The resized and loaded PIL Image.
        """
        image = Image.open(image_path)

        if resize_type == "center_crop_and_resize":
            image = center_crop_and_resize(image, crop_size=crop_size, height=height, width=width)

        elif resize_type == "resize":
            image = image.resize((height, width))

        else:
            raise ValueError("Invalid resize type.")

        return image

    def _setup_generator(self, generator_seed):
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
        image_path: str = "test.png",
        prompts: List[str] = ["A photo of a cat."],
        negative_prompt: List[str] = ["bad"],
        height: int = 512,
        width: int = 512,
        guess_mode: bool = False,
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 20,
        guidance_scale: int = 7.0,
        controlnet_conditioning_scale: int = 1.0,
        generator_seed: int = 0,
        preprocess_type: str = "Canny",
        resize_type: str = "center_crop_and_resize",
        crop_size: int = 512
    ):
        """
        This function generates an image based on the given parameters.

        Args:
        stable_model_path (str): Path to the stable model.
        controlnet_model_path (str): Path to the controlnet model.
        scheduler_name (str): Name of the scheduler.
        image_path (dict): Path of the images.
        prompts (dict): The prompts for image generation.
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
        read_image = self.load_and_resize_image(
            image_path=image_path, resize_type=resize_type, height=height, width=width, crop_size=crop_size)
        

        control_image = preprocces_dicts[preprocess_type](read_image)

        pipe = self.load_model(
            stable_model_path=stable_model_path,
            controlnet_model_path=controlnet_model_path,
            scheduler_name=scheduler_name,
        )

        generator = self._setup_generator(generator_seed)

        output = pipe(
            prompt=prompts,
            height=height,
            width=width,
            guess_mode=guess_mode,
            image=control_image,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            generator=generator,
        ).images

        return output
