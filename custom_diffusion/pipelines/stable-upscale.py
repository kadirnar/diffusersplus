import torch
from diffusers import StableDiffusionUpscalePipeline

from custom_diffusion.pipelines.base import BaseDiffusionModel
from custom_diffusion.utils.data_utils import load_and_resize_image


class StableDiffusionUpscaleGenerator(BaseDiffusionModel):
    """
    A class for generating upscaled images using the stable diffusion model.

    Attributes:
        pipe (StableDiffusionUpscalePipeline): The pipeline for image upscaling.

    Methods:
        _load_diffusion_pipeline: Load the stable diffusion upscale pipeline.
        generate_output: Generate an upscaled image based on the provided parameters.

    Example:
        ```
        upscaler = StableDiffusionUpscaleGenerator()
        upscaled_image = upscaler.generate_output(
            image_path="path_to_low_res_image.png",
            model_path="model_path",
            scheduler_name="DDIM",
            prompt="A clear high-resolution image."
        )
        ```

    """

    def _load_diffusion_pipeline(self, model_path: str = "runwayml/stable-diffusion-v1-5"):
        """
        Load the stable diffusion upscale pipeline.

        Args:
            model_path (str): Path to the upscale diffusion pipeline.
                              Default is "runwayml/stable-diffusion-v1-5".
        """
        self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            safety_checker=None,
            torch_dtype=torch.float16,
            revision="fp16",
        )

    def generate_output(
        self,
        model_path: str = "stabilityai/stable-diffusion-x4-upscaler",
        scheduler_name: str = "DDIM",
        image_path: str = "test.png",
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

        Args:
            model_path (str): Path to the upscale model. Default is "stabilityai/stable-diffusion-x4-upscaler".
            scheduler_name (str): Name of the scheduler. Default is "DDIM".
            image_path (str): Path of the image to be upscaled. Default is "test.png".
            prompt (str): The prompt for image upscaling. Default is "A photo of a cat.".
            negative_prompt (str): The negative prompt for image upscaling. Default is "bad".
            resize_type (str): The type of resizing to apply. Default is "center_crop_and_resize".
            noise_level (int): Noise level for the upscaling process. Default is 20.
            num_images_per_prompt (int): The number of images to generate per prompt. Default is 1.
            num_inference_steps (int): The number of inference steps in image generation. Default is 20.
            guidance_scale (int): The scale of guidance in image generation. Default is 7.0.
            generator_seed (int): The seed for the random generator. Default is 0.

        Returns:
            output (torch.Tensor): The upscaled image.
        """

        read_image = load_and_resize_image(image_path=image_path, resize_type=resize_type, height=512, width=512)

        pipe = self.load_model(model_path=model_path, scheduler_name=scheduler_name)
        generator = self._setup_generator(generator_seed)

        output = pipe(
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
