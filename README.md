<div align="center">
<h2>
     Custom Diffusion: Creating Video from Frame Using Multiple Diffusion
</h2>
<div>
    <a href="https://pepy.tech/project/custom_diffusion"><img src="https://pepy.tech/badge/custom_diffusion" alt="downloads"></a>
    <a href="https://badge.fury.io/py/custom_diffusion"><img src="https://badge.fury.io/py/custom_diffusion.svg" alt="pypi version"></a>
    <a href="https://huggingface.co/spaces/ArtGAN/Stable-Diffusion-ControlNet-WebUI"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg" alt="HuggingFace Spaces"></a>
</div>
</div>


### Installation
```bash
pip install custom_diffusion
```

### Usage
```python

frames_path = video_pipeline(
    video_url="https://huggingface.co/spaces/kadirnar/torchyolo/resolve/main/testv2.mp4",
    youtube=False,
    output_path="output",
    filename="test.mp4",
    quality="720p",
    start_time=0,
    end_time=2,
    frame_rate=1,
)

images_list = load_images_from_folder(frames_path)
image_grid(images_list, rows=5, cols=5)

prompt = "a anime boy"
negative_prompt = "bad"

list_prompt = [prompt] * len(images_list)
list_negative_prompt = [negative_prompt] * len(images_list)

generator = StableDiffusionControlNetGenerator()

generated_image_list = generator.generate_image(
    stable_model_path="andite/anything-v4.0",
    controlnet_model_path="lllyasviel/control_v11p_sd15_canny",
    scheduler_name="EulerAncestralDiscrete",
    images_list=images_list,
    prompt=list_prompt,
    negative_prompt=list_negative_prompt,
    height=512,
    width=512,
    guess_mode=False,
    num_images_per_prompt=1,
    num_inference_steps=30,
    guidance_scale=7.0,
    controlnet_conditioning_scale=1.0,
    generator_seed=0,
    preprocess_type="Canny",
    resize_type="center_crop_and_resize",
    crop_size=512,
)

frame2video = frames_to_video(
    folder_path=generated_image_list,
    output_folder="output",
    output_video_name="frame2video.mp4",
    duration=5,
)

```
