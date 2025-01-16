import cv2

from ccvfi import AutoConfig, AutoModel, BaseConfig, ConfigType
from ccvfi.model import VFIBaseModel

from .util import ASSETS_PATH, calculate_image_similarity, get_device, load_eval_image, load_images


class Test_RIFE:
    def test_official(self) -> None:
        img0, img1, _ = load_images()
        eval_img = load_eval_image()

        for k in [ConfigType.RIFE_IFNet_v426_heavy]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: VFIBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=get_device())
            print(model.device)

            out = model.inference_image_list(img_list=[img0, img1])

            assert len(out) == 1
            for i in range(len(out)):
                cv2.imwrite(str(ASSETS_PATH / f"test_{k}_{i}_out.jpg"), out[i])
                assert calculate_image_similarity(eval_img, out[i])
