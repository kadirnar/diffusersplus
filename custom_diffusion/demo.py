from custom_diffusion.pipelines.controlnet_pipeline import StableDiffusionControlNetGenerator


def main():
    generator = StableDiffusionControlNetGenerator()

    prompts = ["A sunset over a beach"]
    negative_prompt = ["city"]

    generated_image = generator.generate_image(
        stable_model_path="path/to/stable_model",
        controlnet_model_path="path/to/controlnet_model",
        scheduler_name="DDIM",
        image_path="path/to/image.png",
        prompts=prompts,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
        guess_mode=False,
        num_images_per_prompt=1,
        num_inference_steps=100,
        guidance_scale=1.0,
        controlnet_conditioning_scale=1.0,
        generator_seed=123,
        preprocess_type="Canny",
        resize_type="center_crop_and_resize",
    )

    # Save or display your generated image
    # For example, save it as a file
    return generated_image


if __name__ == "__main__":
    main()
