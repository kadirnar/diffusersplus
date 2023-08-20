from typing import List, Optional, Tuple, Union

import torch
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline

from diffusersplus.pipelines.base import BaseDiffusionModel
from diffusersplus.preprocces import preprocces_dicts
from diffusersplus.utils.data_utils import load_and_resize_image


class StableDiffusionXLImageGenerator(BaseDiffusionModel):
    def __init__(
        self,
        stable_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet_model_id: str = "diffusers/controlnet-canny-sdxl-1.0",
        vae_model_id: str = "madebyollin/sdxl-vae-fp16-fix",
        scheduler_name: str = "DDIM",
    ):
        super().__init__()
        self.stable_model_id = stable_model_id
        self.controlnet_model_id = controlnet_model_id
        self.vae_model_id = vae_model_id
        self.scheduler_name = scheduler_name

    def _load_diffusion_pipeline(self):
        """
        Load the stable diffusion pipeline with control net model.
        """
        if not hasattr(self, "pipe") or self.pipe is None:
            self.controlnet = ControlNetModel.from_pretrained(self.controlnet_model_id, torch_dtype=torch.float16)
            self.vae = AutoencoderKL.from_pretrained(self.vae_model_id, torch_dtype=torch.float16)
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                self.stable_model_id, controlnet=self.controlnet, vae=self.vae, torch_dtype=torch.float16
            )
            self.load_scheduler("stable", self.stable_model_id, self.scheduler_name)
            self.pipe.enable_model_cpu_offload()

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image_path: Optional[str] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator_seed: int = 0,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        preprocess_type: str = "Canny",
        resize_type: str = "center_crop_and_resize",
    ) -> torch.Tensor:
        """
        Generate an image based on the provided parameters.
        """

        # Load image and preprocess
        if image_path:
            read_image = load_and_resize_image(
                image_path=image_path, resize_type=resize_type, height=height, width=width
            )
            control_image = preprocces_dicts[preprocess_type](read_image)
        else:
            control_image = None

        # Load model and set up pipeline
        self._load_diffusion_pipeline()
        generator = self._configure_random_generator(generator_seed)

        # Generate the image
        output = self.pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            image=control_image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guess_mode=guess_mode,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            target_size=target_size,
            generator=generator,
        ).images

        return output[0]
