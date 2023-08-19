<div align="center">
<h2>
     Diffusers++: A More User-Friendly Diffusion Library.
</h2> 
<div>
    <a href="https://pypi.org/project/diffusersplus" target="_blank">
        <img src="https://img.shields.io/pypi/pyversions/diffusersplus.svg?color=%2334D058" alt="Supported Python versions">
    </a>
    <a href="https://badge.fury.io/py/diffusersplus"><img src="https://badge.fury.io/py/diffusersplus.svg" alt="pypi version"></a>
    <a href="https://huggingface.co/spaces/ArtGAN/Image-Diffusion-WebUI"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg" alt="HuggingFace Spaces"></a>
</div>
</div>


## Installation
```bash
pip install diffusersplus
```

## Usage
To use the diffusersplus library, follow the steps below for different tasks:

### Stable Diffusion Text2Image Generate:
```python
from diffusersplus import diffusion_pipeline

model = diffusion_pipeline(
    task_id="stable-txt2img", 
    stable_model_id="dreamlike-art/dreamlike-anime-1.0", 
    scheduler_name="DDIM"
)

output = model(
    prompt="A photo of an anime character",
    negative_prompt="bad",
    num_images_per_prompt=1,
    num_inference_steps=30,
    guidance_scale=7.0,
    guidance_rescale=0.0,
    generator_seed=0,
    height=512,
    width=512,
)
```

### Stable Diffusion Image2Image Generate:

```python	
from diffusersplus import diffusion_pipeline

model = diffusion_pipeline(
    task_id="stable-img2img",
    stable_model_id="dreamlike-art/dreamlike-anime-1.0",
    scheduler_name="DDIM"
)

output = model(
    image_path="../data/image.png",
    prompt="A photo of a cat.",
    negative_prompt="bad",
    num_images_per_prompt=1,
    num_inference_steps=50,
    guidance_scale=7.0,
    strength=0.5,
    generator_seed=0,
    resize_type="center_crop_and_resize",
    crop_size=512,
    height=512,
    width=512,
)
```

### Stable Diffusion Upscale:
```python
from diffusersplus import diffusion_pipeline

model = diffusion_pipeline(
    task_id="stable-upscale",
    stable_model_id="stabilityai/stable-diffusion-x4-upscaler",
    scheduler_name="DDIM"
)

output = model(
    image_path="../data/image.png",
    prompt="A photo of a anime character.",
    negative_prompt="bad",
    resize_type="center_crop_and_resize",
    noise_level=20,
    num_images_per_prompt=1,
    num_inference_steps=20,
    guidance_scale=7.0,
    generator_seed=0,
)
```
### Controlnet:
```python
from diffusersplus import diffusion_pipeline

model = diffusion_pipeline(
    task_id="controlnet",
    stable_model_id="dreamlike-art/dreamlike-anime-1.0",
    controlnet_model_id="lllyasviel/sd-controlnet-canny",
    scheduler_name="DDIM",
)
output = model(
    image_path="../data/image.png",
    prompt="A photo of cat.",
    negative_prompt="bad",
    height=512,
    width=512,
    preprocess_type="Canny",
    resize_type="center_crop_and_resize",
    guess_mode=False,
    num_images_per_prompt=1,
    num_inference_steps=50,
    guidance_scale=7.0,
    controlnet_conditioning_scale=0.2,
    generator_seed=0,
)
```

### Controlnet Inpaint:
```python
from diffusersplus import diffusion_pipeline

model = diffusion_pipeline(
    task_id="controlnet-inpaint",
    stable_model_id="dreamlike-art/dreamlike-anime-1.0",
    controlnet_model_id="lllyasviel/sd-controlnet-canny",
    scheduler_name="DDIM",
)
output = model(
    image_path="../data/image.png",
    mask_path="../data/mask_image.png",
    prompt="A photo of a cat.",
    negative_prompt="bad",
    height=512,
    width=512,
    preprocess_type="Canny",
    resize_type="center_crop_and_resize",
    strength=0.5,
    guess_mode=False,
    num_images_per_prompt=1,
    num_inference_steps=50,
    guidance_scale=7.0,
    controlnet_conditioning_scale=1.0,
    generator_seed=0,
)
```

### Controlnet Image2Image:
```python
from diffusersplus import diffusion_pipeline

model = diffusion_pipeline(
    task_id="controlnet-img2img",
    stable_model_id="dreamlike-art/dreamlike-anime-1.0",
    controlnet_model_id="lllyasviel/sd-controlnet-canny",
    scheduler_name="DDIM",
)
output = model(
    image_path="../data/image.png",
    prompt="A photo of a cat.",
    negative_prompt="bad",
    height=512,
    width=512,
    preprocess_type="Canny",
    resize_type="center_crop_and_resize",
    guess_mode=False,
    num_images_per_prompt=1,
    num_inference_steps=20,
    guidance_scale=7.0,
    controlnet_conditioning_scale=1.0,
    strength=0.5,
    generator_seed=0,
)

```
