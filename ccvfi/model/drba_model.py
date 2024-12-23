# type: ignore
from typing import Any, Union

import torch
import torch.nn.functional as F

from ccvfi.arch import DRBA
from ccvfi.model import MODEL_REGISTRY, VFIBaseModel
from ccvfi.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.DRBA)
class DRBAModel(VFIBaseModel):
    def load_model(self) -> Any:
        # cfg: DRBAConfig = self.config
        state_dict = self.get_state_dict()

        HAS_CUDA = True
        try:
            import cupy

            if cupy.cuda.get_cuda_path() is None:
                HAS_CUDA = False
        except Exception:
            HAS_CUDA = False

        model = DRBA(support_cupy=HAS_CUDA)

        model.load_state_dict(self.convert(state_dict), strict=False)
        model.eval().to(self.device)
        if self.fp16:
            model = model.half()
        return model

    def convert(self, param) -> Any:
        return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

    @torch.inference_mode()  # type: ignore
    def inference(
        self,
        Inputs: torch.Tensor,
        minus_t: list[float],
        zero_t: list[float],
        plus_t: list[float],
        left_scene_change: bool,
        right_scene_change: bool,
        scale: float,
        reuse: Any,
    ) -> tuple[tuple[Union[torch.Tensor, torch.Tensor], ...], Any]:
        """
        Inference with the model

        :param Inputs: The input frames (B, 3, C, H, W)
        :param minus_t: Timestep between -1 and 0 (I0 and I1)
        :param zero_t: Timestep of 0, if not empty, preserve I1 (I1)
        :param plus_t: Timestep between 0 and 1 (I1 and I2)
        :param left_scene_change: True if there is a scene change between I0 and I1 (I0 and I1)
        :param right_scene_change: True if there is a scene change between I1 and I2 (I1 and I2)
        :param scale: Flow scale.
        :param reuse: Reusable output from model with last frame pair.

        :return: All immediate frames between I0~I2 and reusable contents.
        """

        def _resize(img: torch.Tensor, _scale: float) -> torch.Tensor:
            _, _, _h, _w = img.shape
            while _h * _scale % 64 != 0:
                _h += 1
            while _w * _scale % 64 != 0:
                _w += 1
            return F.interpolate(img, size=(int(_h), int(_w)), mode="bilinear", align_corners=False)

        def _de_resize(img, ori_h, ori_w) -> torch.Tensor:
            return F.interpolate(img, size=(int(ori_h), int(ori_w)), mode="bilinear", align_corners=False)

        if self.fp16:
            Inputs = Inputs.half()

        I0, I1, I2 = Inputs[:, 0], Inputs[:, 1], Inputs[:, 2]
        _, _, h, w = I0.shape
        I0 = _resize(I0, scale).unsqueeze(0)
        I1 = _resize(I1, scale).unsqueeze(0)
        I2 = _resize(I2, scale).unsqueeze(0)

        inp = torch.cat([I0, I1, I2], dim=1)

        results, reuse = self.model(inp, minus_t, zero_t, plus_t, left_scene_change, right_scene_change, scale, reuse)

        results = tuple(_de_resize(result, h, w) for result in results)

        return results, reuse
