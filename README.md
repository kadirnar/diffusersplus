<div align="center">
<h2>
     Custom Diffusion: Creating Video from Frame Using Diffusion
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

# Importing the required libraries

from custom_diffusion.utils.data_utils import image_grid, load_images_from_folder
from custom_diffusion.pipelines.stable_diffusion_img2img import StableDiffusionImg2ImgGenerator
from custom_diffusion.utils.video_utils import convert_images_to_video, video_pipeline

# Creating a video from a video file
frames_path = video_pipeline(
    video_path="../../data/videos/anime_v0.mp4",
    output_path="../../output",
    start_time=0,
    end_time=2,
    frame_rate=1,
)

# Creating a video from a folder of images
images_list = load_images_from_folder(frames_path, pil_image=False)

prompt = "a anime boy"
negative_prompt = "bad"

list_prompt = [prompt] * len(images_list)
list_negative_prompt = [negative_prompt] * len(images_list)

# Generating images from a list of images
generator = StableDiffusionImg2ImgGenerator()

generated_image_list = generator.generate_image(
    stable_model_path="dreamlike-art/dreamlike-anime-1.0",
    scheduler_name="EulerAncestralDiscrete",
    images_path_list=images_list,
    prompt=list_prompt,
    strength=0.4,
    negative_prompt=list_negative_prompt,
    num_images_per_prompt=1,
    num_inference_steps=50,
    guidance_scale=7.0,
    generator_seed=0,
    resize_type="resize",
)

# Converting the generated images to a video
frame2video = convert_images_to_video(
    image_list=generated_image_list,
    output_file="../../generated_video.mp4",
    frame_rate=10,
)
```
