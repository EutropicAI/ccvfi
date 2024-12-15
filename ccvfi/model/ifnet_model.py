import torch
import cv2
import numpy as np
from torchvision import transforms
from typing import Any
from ccvfi.arch import IFNet
from ccvfi.config import IFNetConfig
from ccvfi.model import MODEL_REGISTRY
from ccvfi.model.vfi_base_model import VFIBaseModel
from ccvfi.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.IFNet)
class IFNetModel(VFIBaseModel):
    def load_model(self) -> Any:
        cfg: IFNetConfig = self.config
        state_dict = self.get_state_dict()

        model = IFNet()

        model.load_state_dict(self.convert(state_dict), strict=False)
        model.eval().to(self.device)
        return model

    def convert(self, param):
        return {
            k.replace("module.", ""): v
            for k, v in param.items()
            if "module." in k
        }

    @torch.inference_mode()  # type: ignore
    def inference(self, img0: np.ndarray, img1: np.ndarray, timestep: float, scale: float) -> np.ndarray:
        """
        Inference with the model

        :param img0: The input image(BGR), can use cv2 to read the image
        :param img1: The input image(BGR), can use cv2 to read the image
        :param timestep: Timestep between 0 and 1 (img0 and img1)
        :param scale: Flow scale.

        :return: an immediate frame between img0 and img1
        """

        def _resize(img, _scale):
            _h, _w, _ = img.shape
            while _h * _scale % 64 != 0:
                _h += 1
            while _w * _scale % 64 != 0:
                _w += 1
            return cv2.resize(img, (_w, _h))

        def _de_resize(img, ori_w, ori_h):
            return cv2.resize(img, (ori_w, ori_h))

        h, w, c = img0.shape
        img0 = _resize(img0, scale)
        img1 = _resize(img1, scale)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img0 = transforms.ToTensor()(img0).unsqueeze(0).to(self.device)
        img1 = transforms.ToTensor()(img1).unsqueeze(0).to(self.device)

        inp = torch.cat([img0, img1], dim=1)
        scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]

        result = self.model(inp, timestep, scale_list)
        result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = (result * 255).clip(0, 255).astype("uint8")

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        result = _de_resize(result, w, h)
        return result
