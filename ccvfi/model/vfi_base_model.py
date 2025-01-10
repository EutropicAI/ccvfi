from typing import Any, List

import numpy as np
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

    @torch.inference_mode()  # type: ignore
    def inference(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()  # type: ignore
    def inference_image_list(self, img_list: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError

    @torch.inference_mode()  # type: ignore
    def inference_video(
        self, clip: Any, scale: float = 1.0, tar_fps: float = 60, scdet: bool = True, scdet_threshold: float = 0.3
    ) -> Any:
        """
        Inference the video with the model, the clip should be a vapoursynth clip

        :param clip: vs.VideoNode
        :param scale: The flow scale factor
        :param tar_fps: The fps of the interpolated video
        :param scdet: Enable SSIM scene change detection
        :param scdet_threshold: SSIM scene change detection threshold (greater is sensitive)
        :return:
        """

        from ccvfi.vs import inference_vfi

        cfg: BaseConfig = self.config

        return inference_vfi(
            inference=self.inference,
            clip=clip,
            scale=scale,
            tar_fps=tar_fps,
            in_frame_count=cfg.in_frame_count,
            scdet=scdet,
            scdet_threshold=scdet_threshold,
            device=self.device,
        )
