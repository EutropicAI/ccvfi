# type: ignore
from typing import Any

import torch
import torch.nn.functional as F

from ccvfi.arch import IFNet
from ccvfi.model import MODEL_REGISTRY
from ccvfi.model.vfi_base_model import VFIBaseModel
from ccvfi.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.IFNet)
class IFNetModel(VFIBaseModel):
    def load_model(self) -> Any:
        state_dict = self.get_state_dict()

        model = IFNet()

        def _convert(param) -> Any:
            return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

        model.load_state_dict(_convert(state_dict), strict=False)
        model.eval().to(self.device)
        return model

    @torch.inference_mode()  # type: ignore
    def inference(self, Inputs: torch.Tensor, timestep: float, scale: float) -> torch.Tensor:
        """
        Inference with the model

        :param Inputs: The input frames (B, 2, C, H, W)
        :param timestep: Timestep between 0 and 1 (img0 and img1)
        :param scale: Flow scale.

        :return: an immediate frame between I0 and I1
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

        I0, I1 = Inputs[:, 0], Inputs[:, 1]
        _, _, h, w = I0.shape
        I0 = _resize(I0, scale)
        I1 = _resize(I1, scale)

        inp = torch.cat([I0, I1], dim=1)
        scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]

        result = self.model(inp, timestep, scale_list)

        result = _de_resize(result, h, w)

        return result
