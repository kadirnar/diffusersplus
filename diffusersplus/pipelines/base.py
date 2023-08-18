from typing import Dict, Union
import torch

from ..utils.scheduler_utils import get_scheduler

class BaseDiffusionModel:
    """
    Base class for diffusion models.
    Provides core functionalities for loading models and generating outputs.
    """

    MODEL_TYPES = ["stable", "controlnet", "pix2pix"]

    def __init__(self, device="cuda"):
        """
        Initialize the BaseDiffusionModel.

        Args:
        - device (str, optional): Device to which the models will be loaded. Defaults to "cuda".
        """
        self.pipe = None
        self.model_cache = {model_type: {} for model_type in self.MODEL_TYPES}
        self.device = device

    def _load_diffusion_pipeline(self, model_path: str):
        """
        Load the diffusion pipeline. To be implemented by derived classes.

        Args:
        - model_path (str): Path to the diffusion model.
        """
        raise NotImplementedError

    def _load_model(self, model_type: str, model_path: str) -> None:
        """
        Helper method to load a model based on its type and path.

        Args:
        - model_type (str): Type of the model (e.g., "stable", "controlnet", "pix2pix").
        - model_path (str): Path to the model.
        """
        if model_path not in self.model_cache[model_type]:
            self._load_diffusion_pipeline(model_path)
            self.model_cache[model_type][model_path] = {"model": self.pipe}

    def load_stable_model(self, model_path: str) -> None:
        """
        Load the stable model.

        Args:
        - model_path (str): Path to the stable model.
        """
        self._load_model("stable", model_path)

    def load_controlnet_model(self, model_path: str) -> None:
        """
        Load the controlnet model.

        Args:
        - model_path (str): Path to the controlnet model.
        """
        self._load_model("controlnet", model_path)

    def load_pix2pix_model(self, model_path: str) -> None:
        """
        Load the pix2pix model.

        Args:
        - model_path (str): Path to the pix2pix model.
        """
        self._load_model("pix2pix", model_path)

    def load_scheduler(self, model_type: str, model_path: str, scheduler_name: str) -> None:
        """
        Load the scheduler for a specific model type and path.

        Args:
        - model_type (str): Type of the model (e.g., "stable", "controlnet", "pix2pix").
        - model_path (str): Path to the model.
        - scheduler_name (str): Name of the scheduler to be loaded.
        """
        model_data = self.model_cache[model_type].get(model_path, {})
        if "scheduler" not in model_data:
            self.pipe = get_scheduler(pipe=self.pipe, scheduler_name=scheduler_name)
            self.pipe.to(self.device)
            self.pipe.enable_xformers_memory_efficient_attention()

            if model_path not in self.model_cache[model_type]:
                self.model_cache[model_type][model_path] = {}

            self.model_cache[model_type][model_path]["scheduler"] = self.pipe

    def _configure_random_generator(self, generator_seed: int = None) -> torch.Generator:
        """
        Configure the random generator with the provided seed.

        Args:
        - generator_seed (int, optional): Seed for the random generator. If None, a random seed will be used.

        Returns:
        - generator (torch.Generator): Configured random generator.
        """
        if generator_seed is 0:
            random_seed = torch.randint(0, 1000000, (1,))
            generator = torch.manual_seed(random_seed)
        else:
            generator = torch.manual_seed(generator_seed)
        return generator

    def _check_model_loaded(self, model_type: str):
        """
        Check if the model of the given type has been loaded.

        Args:
        - model_type (str): Type of the model (e.g., "stable", "controlnet", "pix2pix").
        
        Raises:
        - RuntimeError: If the model of the given type has not been loaded.
        """
        if not self.model_cache[model_type]:
            raise RuntimeError(f"The {model_type} model has not been loaded. Please load the model using the appropriate function.")

    def __call__(self, model_path: str, scheduler_name: str, model_type: str, **kwargs) -> Union[torch.Tensor, Dict]:
        """
        Generate an output based on the provided parameters.

        Args:
        - model_path (str): Path to the model.
        - scheduler_name (str): Name of the scheduler.
        - model_type (str): Type of the model (e.g., "stable", "controlnet", "pix2pix").
        - **kwargs: Additional arguments required by the specific diffusion model.

        Returns:
        - output (Union[torch.Tensor, Dict]): Generated output.
        """
        self._check_model_loaded(model_type)
        # Here, you would continue with the implementation to generate the output based on the model and provided arguments.
        raise NotImplementedError
