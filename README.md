# ccvfi

an inference lib for video frame interpolation with VapourSynth support

### Install

Make sure you have Python >= 3.9 and PyTorch >= 1.13 installed

```bash
pip install ccvfi
```

- Install VapourSynth (optional)

### Start

#### cv2

a simple example to use the RIFE (ECCV2022-RIFE) model to process an image sequence.

```python
import cv2
import numpy as np

from ccvfi import AutoModel, ConfigType, VFIBaseModel

model: VFIBaseModel = AutoModel.from_pretrained(
    pretrained_model_name=ConfigType.RIFE_IFNet_v426_heavy,
)

img0 = cv2.imdecode(np.fromfile("01.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
img1 = cv2.imdecode(np.fromfile("02.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
out = model.inference_image_list(img_list=[img0, img1])[0]
cv2.imwrite("test_out.jpg", out)
```

#### VapourSynth

a simple example to use the VFI (Video Frame-Interpolation) model to process a video (DRBA)

```python
import vapoursynth as vs
from vapoursynth import core

from ccvfi import AutoModel, BaseModelInterface, ConfigType

model: BaseModelInterface = AutoModel.from_pretrained(
    pretrained_model_name=ConfigType.DRBA_IFNet,
)

clip = core.bs.VideoSource(source="s.mp4")
clip = core.resize.Bicubic(clip=clip, matrix_in_s="709", format=vs.RGBH)
clip = model.inference_video(clip, tar_fps=60)
clip = core.resize.Bicubic(clip=clip, matrix_s="709", format=vs.YUV420P16)
clip.set_output()
```

See more examples in the [example](./example) directory, ccvfi can register custom configurations and models to extend the functionality

### Current Support

It still in development, the following models are supported:

- [Architecture](./ccvfi/type/arch.py)

- [Model](./ccvfi/type/model.py)

- [Weight(Config)](./ccvfi/type/config.py)

### Reference

- [PyTorch](https://github.com/pytorch/pytorch)
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
- [mmcv](https://github.com/open-mmlab/mmcv)
- [huggingface transformers](https://github.com/huggingface/transformers)
- [VapourSynth](https://www.vapoursynth.com/)
- [HolyWu's functions](https://github.com/HolyWu)

### License

This project is licensed under the MIT - see
the [LICENSE file](https://github.com/TensoRaws/ccvfi/blob/main/LICENSE) for details.
