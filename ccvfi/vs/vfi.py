import math
from math import exp
from typing import Callable, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
import vapoursynth as vs
from torch import Tensor
from vapoursynth import core

from ccvfi.vs.convert import frame_to_tensor, tensor_to_frame


class TMapper:
    def __init__(self, src: float = -1.0, dst: float = 0.0, times: float = -1):
        self.times = dst / src if times == -1 else times
        self.now_step = -1
        self.src = src
        self.dst = dst

    def get_range_timestamps(
        self, _min: float, _max: float, lclose: bool = True, rclose: bool = False, normalize: bool = True
    ) -> list:
        _min_step = math.ceil(_min * self.times)
        _max_step = math.ceil(_max * self.times)
        _start = _min_step if lclose else _min_step + 1
        _end = _max_step if not rclose else _max_step + 1
        if _start >= _end:
            return []
        if normalize:
            return [((_i / self.times) - _min) / (_max - _min) for _i in range(_start, _end)]
        return [_i / self.times for _i in range(_start, _end)]


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window_3d(window_size: int, channel: int = 1) -> torch.Tensor:
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous()
    return window


def ssim_matlab(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    window: torch.Tensor = None,
    size_average: bool = True,
    full: bool = False,
) -> torch.Tensor:
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if torch.max(img1) > 128:
        max_val = 255
    else:
        max_val = 1

    if torch.min(img1) < -0.5:
        min_val = -1
    else:
        min_val = 0
    L = max_val - min_val

    padd = 0
    (_, _, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(real_size, channel=1).to(img1.device)
        # Channel is set to 1 since we consider color images as volumetric images

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode="replicate"), window, padding=padd, groups=1)
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode="replicate"), window, padding=padd, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), "replicate"), window, padding=padd, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), "replicate"), window, padding=padd, groups=1) - mu2_sq
    sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), "replicate"), window, padding=padd, groups=1) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def check_scene(x1: torch.Tensor, x2: torch.Tensor, enable_scdet: bool, scdet_threshold: float) -> Union[bool, Tensor]:
    if not enable_scdet:
        return False
    x1 = F.interpolate(x1[0].clone().float(), (32, 32), mode="bilinear", align_corners=False)
    x2 = F.interpolate(x2[0].clone().float(), (32, 32), mode="bilinear", align_corners=False)
    return ssim_matlab(x1, x2) < scdet_threshold


def inference_vfi(
    inference: Callable,
    clip: vs.VideoNode,
    scale: float,
    tar_fps: float,
    device: torch.device,
    in_frame_count: int = 2,
    scdet: bool = True,
    scdet_threshold: float = 0.3,
) -> vs.VideoNode:
    """
    Inference the video with the model, the clip should be a vapoursynth clip

    :param inference: The inference function
    :param clip: vs.VideoNode
    :param scale: The flow scale factor
    :param tar_fps: The fps of the interpolated video
    :param device: The device
    :param in_frame_count: The input frame count of vfi method once infer
    :param scdet: Enable SSIM scene change detection
    :param scdet_threshold: SSIM scene change detection threshold (greater is sensitive)
    :return:
    """

    if core.num_threads != 1:
        raise ValueError("The number of threads must be 1 when enable frame interpolation")

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("Only vs.RGBH and vs.RGBS formats are supported")

    if clip.num_frames < in_frame_count:
        raise ValueError(f"Clip do not have enough frames for vfi method require {in_frame_count} frames once infer")

    src_fps = clip.fps.numerator / clip.fps.denominator
    if src_fps > tar_fps:
        raise ValueError("The target fps should be greater than the clip fps")

    if scale < 0 or not math.log2(scale).is_integer():
        raise ValueError("The scale should be greater than 0 and is power of two")

    vfi_methods = {
        2: inference_vsr_two_frame_in,
        3: inference_vsr_three_frame_in,
    }

    if in_frame_count not in vfi_methods:
        raise ValueError(f"The vfi method with {in_frame_count} frame input is not supported")

    mapper = TMapper(src_fps, tar_fps)

    return vfi_methods[in_frame_count](inference, clip, mapper, scale, scdet, scdet_threshold, device)


def inference_vsr_two_frame_in(
    inference: Callable,
    clip: vs.VideoNode,
    mapper: TMapper,
    scale: float,
    scdet: bool,
    scdet_threshold: float,
    device: torch.device,
) -> vs.VideoNode:
    """
    VFI for two frame input models

    f1, f2 -> f1?, f1t?, f2?

    For the two frame input model, the inference function should accept a tensor with shape (b, 2, c, h, w)
    And return a tensor with shape (b, c, h, w)

    :param inference: The inference function
    :param clip: vs.VideoNode
    :param scale: The flow scale factor
    :param mapper: The framerate mapper
    :param scdet: Enable SSIM scene change detection
    :param scdet_threshold: SSIM scene change detection threshold (greater is sensitive)
    :param device: The device
    :return:
    """

    in_idx: int = 0
    out_idx: int = 0
    in_frames: Dict[int, torch.Tensor] = {}
    out_frames: Dict[int, torch.Tensor] = {}
    flag_end: bool = False
    reuse: tuple[torch.Tensor, ...]

    def to_input_tensor(x: vs.VideoFrame) -> torch.Tensor:
        return frame_to_tensor(x, device=device).unsqueeze(0).unsqueeze(0)

    new_clip = clip.std.AssumeFPS(fpsnum=mapper.dst, fpsden=1)
    less_num_frames = math.ceil(clip.num_frames * mapper.dst / mapper.src) - clip.num_frames
    for _ in range(less_num_frames):
        new_clip = new_clip.std.DuplicateFrames(clip.num_frames - 1)

    def _inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        nonlocal in_idx, out_idx, in_frames, out_frames, flag_end, reuse
        if n >= out_idx and not flag_end:
            if in_idx not in in_frames.keys():
                in_frames[in_idx] = to_input_tensor(clip.get_frame(in_idx))
            I0 = in_frames[in_idx]

            if in_idx + 1 >= clip.num_frames - 1:
                flag_end = True
                return tensor_to_frame(out_frames[list(out_frames.keys())[-1]], f[1].copy())

            if in_idx + 1 not in in_frames.keys():
                in_frames[in_idx + 1] = to_input_tensor(clip.get_frame(in_idx + 1))
            I1 = in_frames[in_idx + 1]

            ts = mapper.get_range_timestamps(in_idx, in_idx + 1, lclose=True, rclose=flag_end, normalize=True)

            scene = check_scene(I0, I1, scdet, scdet_threshold)

            for t in ts:
                if scene:
                    out = I0.squeeze(0)
                else:
                    if t == 0:
                        out = I0.squeeze(0)
                    elif t == 1:
                        out = I1.squeeze(0)
                    else:
                        out = inference(torch.cat([I0, I1], dim=1), timestep=t, scale=scale)
                out_frames[out_idx] = out
                out_idx += 1

            # clear input cache
            if in_idx - 1 in in_frames.keys():
                in_frames.pop(in_idx - 1)

            in_idx += 1

        # clear output cache
        if n - 1 in out_frames.keys() and len(out_frames.keys()) > 2:
            out_frames.pop(n - 1)

        if n not in out_frames.keys():
            return tensor_to_frame(out_frames[list(out_frames.keys())[-1]], f[1].copy())

        return tensor_to_frame(out_frames[n], f[1].copy())

    return new_clip.std.ModifyFrame([new_clip, new_clip], _inference)


def inference_vsr_three_frame_in(
    inference: Callable,
    clip: vs.VideoNode,
    mapper: TMapper,
    scale: float,
    scdet: bool,
    scdet_threshold: float,
    device: torch.device,
) -> vs.VideoNode:
    """
    VFI for three frame input models

    f1, f2, f3 -> f1?, f1t?, f2?, f2t?, f3?

    For the three frame input model, the inference function should accept a tensor with shape (b, 3, c, h, w)
    And return a tensor with shape (b, c, h, w)

    :param inference: The inference function
    :param clip: vs.VideoNode
    :param scale: The flow scale factor
    :param mapper: The framerate mapper
    :param scdet: Enable SSIM scene change detection
    :param scdet_threshold: SSIM scene change detection threshold (greater is sensitive)
    :param device: The device
    :return:
    """

    in_idx: int = 0
    out_idx: int = 0
    in_frames: Dict[int, torch.Tensor] = {}
    out_frames: Dict[int, torch.Tensor] = {}
    flag_end: bool = False
    reuse: tuple[torch.Tensor, ...]

    def calc_t(_mapper: TMapper, _idx: float, _flag_end: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ts = _mapper.get_range_timestamps(_idx - 0.5, _idx + 0.5, lclose=True, rclose=_flag_end, normalize=False)
        timestamp = np.asarray(ts, dtype=float) - _idx
        vfi_timestamp = np.round(timestamp, 4)

        minus_t = vfi_timestamp[vfi_timestamp < 0]
        zero_t = vfi_timestamp[vfi_timestamp == 0]
        plus_t = vfi_timestamp[vfi_timestamp > 0]
        return minus_t, zero_t, plus_t

    def to_input_tensor(x: vs.VideoFrame) -> torch.Tensor:
        return frame_to_tensor(x, device=device).unsqueeze(0).unsqueeze(0)

    new_clip = clip.std.AssumeFPS(fpsnum=mapper.dst, fpsden=1)
    less_num_frames = math.ceil(clip.num_frames * mapper.dst / mapper.src) - clip.num_frames
    for _ in range(less_num_frames):
        new_clip = new_clip.std.DuplicateFrames(clip.num_frames - 1)

    def _inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        nonlocal in_idx, out_idx, in_frames, out_frames, flag_end, reuse
        if n >= out_idx and not flag_end:
            if in_idx not in in_frames.keys():
                in_frames[in_idx] = to_input_tensor(clip.get_frame(in_idx))
            I0 = in_frames[in_idx]

            if in_idx + 1 >= clip.num_frames - 1:
                flag_end = True
                return tensor_to_frame(out_frames[list(out_frames.keys())[-1]], f[1].copy())

            if in_idx + 1 not in in_frames.keys():
                in_frames[in_idx + 1] = to_input_tensor(clip.get_frame(in_idx + 1))
            I1 = in_frames[in_idx + 1]

            if in_idx + 2 >= clip.num_frames - 1:
                flag_end = True
            else:
                if in_idx + 2 not in in_frames.keys():
                    in_frames[in_idx + 2] = to_input_tensor(clip.get_frame(in_idx + 2))
                I2 = in_frames[in_idx + 2]

            mt, zt, pt = calc_t(mapper, in_idx, flag_end)
            left_scene = check_scene(I0, I1, scdet, scdet_threshold)
            if in_idx == 0:  # head
                right_scene = left_scene
                output, reuse = inference(torch.cat([I0, I0, I1], dim=1), mt, zt, pt, False, right_scene, scale, None)
            elif flag_end:  # tail
                output, _ = inference(torch.cat([I0, I1, I1], dim=1), mt, zt, pt, left_scene, False, scale, reuse)
            else:
                right_scene = check_scene(I1, I2, scdet, scdet_threshold)
                output, reuse = inference(
                    torch.cat([I0, I1, I2], dim=1), mt, zt, pt, left_scene, right_scene, scale, reuse
                )

            for x in output:
                out_frames[out_idx] = x
                out_idx += 1

            # clear input cache
            if in_idx - 1 in in_frames.keys():
                in_frames.pop(in_idx - 1)

            in_idx += 1

        # clear output cache
        if n - 1 in out_frames.keys() and len(out_frames.keys()) > 2:
            out_frames.pop(n - 1)

        if n not in out_frames.keys():
            return tensor_to_frame(out_frames[list(out_frames.keys())[-1]], f[1].copy())

        return tensor_to_frame(out_frames[n], f[1].copy())

    return new_clip.std.ModifyFrame([new_clip, new_clip], _inference)
