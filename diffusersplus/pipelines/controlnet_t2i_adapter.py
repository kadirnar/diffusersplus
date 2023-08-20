from typing import List, Optional, Union

import torch
from diffusers import StableDiffusionAdapterPipeline, T2IAdapter
from PIL import Image

from diffusersplus.pipelines.base import BaseDiffusionModel
from diffusersplus.preprocces import preprocces_dicts
from diffusersplus.utils.data_utils import load_and_resize_image


class StableDiffusionT2iAdapterGenerator(BaseDiffusionModel):
    """
    A class to handle image generation using stable diffusion with preprocessing.
    """

    def __init__(
        self,
        stable_model_id: str = "runwayml/stable-diffusion-v1-5",
        adapter_model_id: str = "TencentARC/t2iadapter_canny_sd15v2",
        controlnet_model_id: str = None,
        vae_model_id: str = None,
        scheduler_name: str = "DDIM",
    ):
        super().__init__()
        self.vae_model_id = vae_model_id
        self.stable_model_id = stable_model_id
        self.controlnet_model_id = controlnet_model_id
        self.adapter_model_id = adapter_model_id
        self.scheduler_name = scheduler_name

    def _load_diffusion_pipeline(self):
        """
        Load the stable diffusion pipeline specific to preprocessing.
        """
        if not hasattr(self, "pipe") or self.pipe is None:
            adapter = T2IAdapter.from_pretrained(self.adapter_model_id, torch_dtype=torch.float16)
            self.pipe = StableDiffusionAdapterPipeline.from_pretrained(
                self.stable_model_id,
                adapter=adapter,
                torch_dtype=torch.float16,
            )
            self.load_scheduler("stable", self.stable_model_id, self.scheduler_name)

    def __call__(
        self,
        image_path: str,
        prompt: str = "A photo of a cat.",
        negative_prompt: str = "bad",
        height: int = 512,
        width: int = 512,
        preprocess_type: str = "Canny",
        resize_type: str = "center_crop_and_resize",
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 20,
        guidance_scale: int = 7.0,
        adapter_conditioning_scale: int = 1.0,
        generator_seed: int = 0,
    ) -> torch.Tensor:
        """
        Generate an image based on the provided parameters.
        """

        # Load the model
        self._load_diffusion_pipeline()

        # Load image and preprocess
        read_image = load_and_resize_image(image_path=image_path, resize_type=resize_type, height=height, width=width)
        control_image = preprocces_dicts[preprocess_type](read_image)

        generator = self._configure_random_generator(generator_seed)

        # Generate the image
        output = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            image=control_image,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            adapter_conditioning_scale=float(adapter_conditioning_scale),
            generator=generator,
        ).images

        return output
