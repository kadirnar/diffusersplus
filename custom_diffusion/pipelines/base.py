from typing import Dict, List, Optional, Union

import torch
from diffusers import StableDiffusionPipeline

from custom_diffusion.utils.scheduler_utils import get_scheduler


class BaseDiffusionModel:
    """
    Base class for diffusion models.
    Provides core functionalities for loading models and generating outputs.
    """

    def __init__(self):
        self.pipe = None
        self.model_cache = {}

    def _load_diffusion_pipeline(self, model_path: str):
        """
        Load the diffusion pipeline. To be implemented by derived classes.

        Args:
        - model_path (str): Path to the diffusion model.
        """
        raise NotImplementedError

    def load_model(self, model_path: str, scheduler_name: str) -> StableDiffusionPipeline:
        """
        Load the models and setup the scheduler if not cached.

        Args:
        - model_path (str): Path to the model.
        - scheduler_name (str): Name of the scheduler.

        Returns:
        - pipe (StableDiffusionPipeline): Configured model pipeline.
        """
        cache_key = (model_path, scheduler_name)

        # Load and setup models only if they're not in the cache
        if cache_key not in self.model_cache:
            self._load_diffusion_pipeline(model_path)
            self.pipe = get_scheduler(pipe=self.pipe, scheduler_name=scheduler_name)
            self.pipe.to("cuda")
            self.pipe.enable_xformers_memory_efficient_attention()

            self.model_cache[cache_key] = self.pipe

        return self.model_cache[cache_key]

    def _setup_generator(self, generator_seed: int = 0) -> torch.Generator:
        """
        Setup the random generator with the provided seed.

        Args:
        - generator_seed (int): Seed for the random generator.

        Returns:
        - generator (torch.Generator): Configured random generator.
        """
        if generator_seed == 0:
            random_seed = torch.randint(0, 1000000, (1,))
            generator = torch.manual_seed(random_seed)
        else:
            generator = torch.manual_seed(generator_seed)
        return generator

    def generate_output(self, model_path: str, scheduler_name: str, **kwargs) -> Union[torch.Tensor, Dict]:
        """
        Generate an output based on the provided parameters. To be implemented by derived classes.

        Args:
        - model_path (str): Path to the model.
        - scheduler_name (str): Name of the scheduler.
        - **kwargs: Additional arguments required by the specific diffusion model.

        Returns:
        - output (Union[torch.Tensor, Dict]): Generated output.
        """
        raise NotImplementedError
