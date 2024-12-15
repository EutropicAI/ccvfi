# type: ignore
import torch
import cv2
import numpy as np
from numpy import ndarray
from torchvision import transforms
from typing import Any, Tuple, Union
from ccvfi.arch import DRBA
from ccvfi.config import DRBAConfig
from ccvfi.model import MODEL_REGISTRY
from ccvfi.model import VFIBaseModel
from ccvfi.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.DRBA)
class DRBAModel(VFIBaseModel):

    def load_model(self) -> Any:
        cfg: DRBAConfig = self.config
        state_dict = self.get_state_dict()

        HAS_CUDA = True
        try:
            import cupy
            if cupy.cuda.get_cuda_path() == None:
                HAS_CUDA = False
        except Exception:
            HAS_CUDA = False

        model = DRBA(
            support_cupy=HAS_CUDA
        )

        model.load_state_dict(self.convert(state_dict), strict=False)
        model.eval().to(self.device)
        return model

    def convert(self, param) -> Any:
        return {
            k.replace("module.", ""): v
            for k, v in param.items()
            if "module." in k
        }

    @torch.inference_mode()  # type: ignore
    def inference(self, img0: np.ndarray, img1: np.ndarray, img2: np.ndarray,
                  minus_t: list[float], zero_t: list[float], plus_t: list[float],
                  left_scene_change: bool, right_scene_change: bool, scale: float,
                  reuse: Any) -> tuple[tuple[Union[ndarray, ndarray], ...], Any]:
        """
        Inference with the model

        :param img0: The input image(BGR), can use cv2 to read the image
        :param img1: The input image(BGR), can use cv2 to read the image
        :param img2: The input image(BGR), can use cv2 to read the image
        :param minus_t: Timestep between -1 and 0 (img0 and img1)
        :param zero_t: Timestep of 0, if not empty, preserve img1 (img1)
        :param plus_t: Timestep between 0 and 1 (img1 and img2)
        :param plus_t: Timestep between 0 and 1 (img1 and img2)
        :param left_scene_change: True if there is a scene change between img0 and img1 (img0 and img1)
        :param right_scene_change: True if there is a scene change between img1 and img2 (img1 and img2)
        :param scale: Flow scale.
        :param reuse: Reusable output from model with last frame pair.

        :return: All immediate frames between img0~img2(May contain img0, img1, img2) and reusable contents.
        """

        def _resize(img, _scale) -> np.ndarray:
            _h, _w, _ = img.shape
            while _h * _scale % 64 != 0:
                _h += 1
            while _w * _scale % 64 != 0:
                _w += 1
            return cv2.resize(img, (_w, _h))

        def _de_resize(img, ori_w, ori_h) -> np.ndarray:
            return cv2.resize(img, (ori_w, ori_h))

        h, w, c = img0.shape
        img0 = _resize(img0, scale)
        img1 = _resize(img1, scale)
        img2 = _resize(img2, scale)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img0 = transforms.ToTensor()(img0).unsqueeze(0).to(self.device)
        img1 = transforms.ToTensor()(img1).unsqueeze(0).to(self.device)
        img2 = transforms.ToTensor()(img2).unsqueeze(0).to(self.device)

        results, reuse = self.model(img0, img1, img2, minus_t, zero_t, plus_t, left_scene_change, right_scene_change,
                                    scale, reuse)

        def _convert(result) -> np.ndarray:
            result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
            result = (result * 255).clip(0, 255).astype("uint8")

            return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        results = tuple(_convert(result) for result in results)
        results = tuple(_de_resize(result, w, h) for result in results)

        return results, reuse
