import cv2
import numpy as np
import torch

from ccvfi import AutoConfig, AutoModel, BaseConfig, ConfigType
from ccvfi.model import VFIBaseModel

from .util import ASSETS_PATH, calculate_image_similarity, get_device, load_eval_image, load_images


class Test_IFNet:
    def test_official(self) -> None:
        img0, img1, img2 = load_images()
        eval_img = load_eval_image()

        for k in [ConfigType.IFNet_v426_heavy]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: VFIBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=get_device())
            print(model.device)

            I0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).to(model.device) / 255.0
            I1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(model.device) / 255.0
            I0 = I0.unsqueeze(0)
            I1 = I1.unsqueeze(0)
            inp = torch.cat([I0, I1], dim=1)

            out = model.inference(inp, timestep=0.5, scale=1.0)
            imgOut = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            cv2.imwrite(str(ASSETS_PATH / "test_out.jpg"), imgOut)

            assert calculate_image_similarity(eval_img, imgOut)
