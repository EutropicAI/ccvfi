import cv2
import numpy as np
import torch

from ccvfi import AutoConfig, AutoModel, BaseConfig, ConfigType
from ccvfi.model import VFIBaseModel

from .util import ASSETS_PATH, calculate_image_similarity, get_device, load_eval_images, load_images


class Test_DRBA:
    def test_official(self) -> None:
        img0, img1, img2 = load_images()
        eval_imgs = load_eval_images()

        for k in [ConfigType.DRBA_IFNet]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: VFIBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=get_device())
            print(model.device)

            I0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).float().to(model.device) / 255.0
            I1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(model.device) / 255.0
            I2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(model.device) / 255.0
            I0 = I0.unsqueeze(0)
            I1 = I1.unsqueeze(0)
            I2 = I2.unsqueeze(0)
            inp = torch.cat([I0, I1, I2], dim=1)

            Outputs, _ = model.inference(inp, [-1, -0.5], [0], [0.5, 1], False, False, 1.0, None)

            for i in range(len(Outputs)):
                out = (Outputs[i].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(str(ASSETS_PATH / f"test_out_{i}.jpg"), out)

                assert calculate_image_similarity(eval_imgs[i], out)
