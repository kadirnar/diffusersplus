import torch
from diffusers import StableDiffusionPipeline

from ..pipelines.base import BaseDiffusionModel


class StableDiffusionText2ImgGenerator(BaseDiffusionModel):
    """
    A class to handle image generation using stable diffusion models for text-to-image tasks.
    Inherits from the BaseDiffusionModel to utilize core functionalities.

    Example:
    ```python
    generator = StableDiffusionText2ImgGenerator(stable_model_id="runwayml/stable-diffusion-v1-5", scheduler_name="DDIM")
    output = generator(prompt=["A beautiful landscape with mountains."])
    ```
    """

    def __init__(self, stable_model_id: str = None, controlnet_model_id: str = None, scheduler_name=None):
        super().__init__()
        self.stable_model_id = stable_model_id
        self.controlnet_model_id = controlnet_model_id
        self.scheduler_name = scheduler_name

    def _load_diffusion_pipeline(self):
        """
        Load the stable diffusion pipeline specific to text-to-image generation.
        """
        self.pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=self.stable_model_id,
            safety_checker=None,
            torch_dtype=torch.float16,
        )

    def __call__(
        self,
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
        ... [Rest of the docstring]
        """

        # Check if the model is already loaded in cache, else load it
        if self.stable_model_id not in self.model_cache["stable"]:
            self._load_diffusion_pipeline()
            self.load_scheduler("stable", self.stable_model_id, self.scheduler_name)

        generator = self._configure_random_generator(generator_seed)

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
