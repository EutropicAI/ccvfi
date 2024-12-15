import cv2

from ccvfi import AutoConfig, AutoModel, BaseConfig, ConfigType
from ccvfi.model import VFIBaseModel

from .util import ASSETS_PATH, calculate_image_similarity, get_device, load_images


class Test_DRBA:
    def test_official(self) -> None:
        img0, img1, img2 = load_images()

        for k in [ConfigType.DRBA_IFNet]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: VFIBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=get_device())
            print(model.device)

            imgOutputs, _ = model.inference(img0, img1, img2, [-1, -0.5], [0], [0.5, 1]
                                            , False, False, 1.0, None)

            for i in range(len(imgOutputs)):
                cv2.imwrite(str(ASSETS_PATH / f"test_out_{i}.jpg"), imgOutputs[i])

                assert calculate_image_similarity(img0, imgOutputs[i])
