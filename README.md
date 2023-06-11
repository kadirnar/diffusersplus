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
from custom_diffusion.pipelines.controlnet_pipeline import StableDiffusionControlNetGenerator


generator = StableDiffusionControlNetGenerator()

generated_image = generator.generate_image(
     stable_model_path="runwayml/stable-diffusion-v1-5"
     controlnet_model_path="lllyasviel/control_v11p_sd15_canny",
     scheduler_name="DDIM",
     image_path="test.png",
     prompt="Anime boy",
     negative_prompt="bad",
     height=512,
     width=512,
     guess_mode=False,
     num_images_per_prompt=1,
     num_inference_steps=20,
     guidance_scale=7.0,
     controlnet_conditioning_scale=1.0,
     generator_seed=0,
     preprocess_type="Canny",
     resize_type="center_crop_and_resize",
     crop_size=512,
)
```
