def main():
    generator = StableDiffusionControlNetGenerator()


    generated_image = generator.generate_image(
        stable_model_path="runwayml/stable-diffusion-v1-5",
        controlnet_model_path="lllyasviel/sd-controlnet-scribble",
        scheduler_name="DDIM",
        image_path="data/1.png",
        prompts="anime girl",
        negative_prompt="bad",
        height=512,
        width=512,
        guess_mode=False,
        num_images_per_prompt=1,
        num_inference_steps=10,
        guidance_scale=7.0,
        controlnet_conditioning_scale=1.0,
        generator_seed=0,
        preprocess_type="Canny",
        resize_type="center_crop_and_resize",
        crop_size=512,
    )

    # Save or display your generated image
    # For example, save it as a file
    return generated_image


if __name__ == "__main__":
    output = main()
