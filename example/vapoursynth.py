import sys

sys.path.append(".")
sys.path.append("..")

import vapoursynth as vs
from vapoursynth import core

from ccvfi import AutoModel, BaseModelInterface, ConfigType

# --- IFNet, use fp16 to inference (vs.RGBH)

model: BaseModelInterface = AutoModel.from_pretrained(
    pretrained_model_name=ConfigType.IFNet_v426_heavy,
)

core.num_threads = 1  # 设置为单线程
clip = core.bs.VideoSource(source="./video/test.mp4")
clip = core.resize.Bicubic(clip=clip, matrix_in_s="709", format=vs.RGBH)
clip = model.inference_video(clip, scale=1.0, tar_fps=120)
clip = core.resize.Bicubic(clip=clip, matrix_s="709", format=vs.YUV420P16)
clip.set_output()

# ---  use fp32 to inference (vs.RGBS)

# model: BaseModelInterface = AutoModel.from_pretrained(
#     pretrained_model_name=ConfigType.IFNet_v426_heavy,
#     fp16 = False
# )

# core.num_threads = 1  # 设置为单线程
# clip = core.bs.VideoSource(source="./video/test.mp4")
# clip = core.resize.Bicubic(clip=clip, matrix_in_s="709", format=vs.RGBH)
# clip = model.inference_video(clip, scale=1.0, tar_fps=120)
# clip = core.resize.Bicubic(clip=clip, matrix_s="709", format=vs.YUV420P16)
# clip.set_output()
