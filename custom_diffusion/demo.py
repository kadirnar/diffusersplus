from custom_diffusion.pipelines.controlnet_pipeline import StableDiffusionControlNetGenerator


def main(
    stable_model_path: str = "runwayml/stable-diffusion-v1-5",
    controlnet_model_path: str = "lllyasviel/control_v11p_sd15_canny",
    scheduler_name: str = "DDIM",
    image_path: str = "test.png",
    prompt: str = "Anime boy",
    negative_prompt: str = "bad",
    height: int = 512,
    width: int = 512,
    guess_mode: bool = False,
    num_images_per_prompt: int = 1,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.0,
    controlnet_conditioning_scale: float = 1.0,
    generator_seed: int = 0,
    preprocess_type: str = "Canny",
    resize_type: str = "center_crop_and_resize",
    crop_size: int = 512,
):
    generator = StableDiffusionControlNetGenerator()

    generated_image = generator.generate_image(
        stable_model_path=stable_model_path,
        controlnet_model_path=controlnet_model_path,
        scheduler_name=scheduler_name,
        image_path=image_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        guess_mode=guess_mode,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator_seed=generator_seed,
        preprocess_type=preprocess_type,
        resize_type=resize_type,
        crop_size=crop_size,
    )

    return generated_image
