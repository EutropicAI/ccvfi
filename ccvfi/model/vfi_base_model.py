from typing import Any

import torch

from ccvfi.cache_models import load_file_from_url
from ccvfi.type import BaseConfig, BaseModelInterface


class VFIBaseModel(BaseModelInterface):
    def get_state_dict(self) -> Any:
        """
        Load the state dict of the model from config

        :return: The state dict of the model
        """
        cfg: BaseConfig = self.config

        if cfg.path is not None:
            state_dict_path = str(cfg.path)
        else:
            try:
                state_dict_path = load_file_from_url(
                    config=cfg, force_download=False, model_dir=self.model_dir, gh_proxy=self.gh_proxy
                )
            except Exception as e:
                print(f"Error: {e}, try force download the model...")
                state_dict_path = load_file_from_url(
                    config=cfg, force_download=True, model_dir=self.model_dir, gh_proxy=self.gh_proxy
                )

        return torch.load(state_dict_path, map_location=self.device, weights_only=True)
