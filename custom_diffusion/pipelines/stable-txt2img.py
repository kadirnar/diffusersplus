import torch
from diffusers import StableDiffusionPipeline

from custom_diffusion.pipelines.base import BaseDiffusionModel


class StableDiffusionText2ImgGenerator(BaseDiffusionModel):
    """
    A class to handle image generation using stable diffusion models for text-to-image tasks.
    Inherits from the BaseDiffusionModel to utilize core functionalities.

    Example:
    ```python
    generator = StableDiffusionText2ImgGenerator()
    output = generator.generate_output(
        model_path="runwayml/stable-diffusion-v1-5",
        scheduler_name="DDIM",
        prompt=["A beautiful landscape with mountains."]
    )
    ```
    """

    def _load_diffusion_pipeline(self, model_path: str):
        """
        Load the stable diffusion pipeline specific to text-to-image generation.

        Args:
        - model_path (str): Path to the stable diffusion pipeline.
        """
        # Check if the model is already cached
        if model_path in self.model_cache:
            self.pipe = self.model_cache[model_path]
            return

        self.pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            safety_checker=None,
            torch_dtype=torch.float16,
        )
        self.model_cache[model_path] = self.pipe

    def generate_output(
        self,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        scheduler_name: str = "DDIM",
        prompt: str = "A photo of a anime boy.",
        negative_prompt: str = "bad",
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: int = 7.0,
        guidance_rescale: float = 0.0,
        generator_seed: int = 0,
        height: int = 512,
        width: int = 512,
    ) -> torch.Tensor:
        """
        Generate an image based on the provided parameters.

        Args:
        - model_path (str): Path to the stable model.
        - scheduler_name (str): Name of the scheduler.
        - prompt (List[str]): The prompt for image generation.
        - negative_prompt (List[str]): The negative prompt for image generation.
        - height (int): The height of the image to generate.
        - width (int): The width of the image to generate.
        - num_images_per_prompt (int): The number of images to generate per prompt.
        - num_inference_steps (int): The number of inference steps in image generation.
        - guidance_scale (int): The scale of guidance in image generation.
        - guidance_rescale (float): The rescale value for guidance in image generation.
        - generator_seed (int): The seed for the random generator.

        Returns:
        - output (torch.Tensor): The generated image.
        """
        self.load_model(model_path=model_path, scheduler_name=scheduler_name)

        generator = self._setup_generator(generator_seed)

        output = self.pipe(
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
