import cv2

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

            imgOut = model.inference(img0, img1, timestep=0.5, scale=1.0)

            cv2.imwrite(str(ASSETS_PATH / "test_out.jpg"), imgOut)

            assert calculate_image_similarity(eval_img, imgOut)
