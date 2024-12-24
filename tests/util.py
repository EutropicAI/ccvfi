import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

from ccvfi.util.device import DEFAULT_DEVICE

print(f"PyTorch version: {torch.__version__}")
torch_2_4: bool = torch.__version__.startswith("2.4")

ASSETS_PATH = Path(__file__).resolve().parent.parent.absolute() / "assets"
TEST_IMG_PATH0 = ASSETS_PATH / "test_i0.png"
TEST_IMG_PATH1 = ASSETS_PATH / "test_i1.png"
TEST_IMG_PATH2 = ASSETS_PATH / "test_i2.png"
EVAL_IMG_PATH = ASSETS_PATH / "test_out.jpg"
EVAL_IMG_PATH0 = ASSETS_PATH / "test_out_0.jpg"
EVAL_IMG_PATH1 = ASSETS_PATH / "test_out_1.jpg"
EVAL_IMG_PATH2 = ASSETS_PATH / "test_out_2.jpg"
EVAL_IMG_PATH3 = ASSETS_PATH / "test_out_3.jpg"
EVAL_IMG_PATH4 = ASSETS_PATH / "test_out_4.jpg"


def get_device() -> torch.device:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return torch.device("cpu")
    return DEFAULT_DEVICE


def load_images() -> tuple[np.ndarray, ...]:
    img0 = cv2.imdecode(np.fromfile(str(TEST_IMG_PATH0), dtype=np.uint8), cv2.IMREAD_COLOR)
    img1 = cv2.imdecode(np.fromfile(str(TEST_IMG_PATH1), dtype=np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.fromfile(str(TEST_IMG_PATH2), dtype=np.uint8), cv2.IMREAD_COLOR)
    img0 = cv2.resize(img0, (960, 540))
    img1 = cv2.resize(img1, (960, 540))
    img2 = cv2.resize(img2, (960, 540))

    return img0, img1, img2


def load_eval_images() -> tuple[np.ndarray, ...]:
    img0 = cv2.imdecode(np.fromfile(str(EVAL_IMG_PATH0), dtype=np.uint8), cv2.IMREAD_COLOR)
    img1 = cv2.imdecode(np.fromfile(str(EVAL_IMG_PATH1), dtype=np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.fromfile(str(EVAL_IMG_PATH2), dtype=np.uint8), cv2.IMREAD_COLOR)
    img3 = cv2.imdecode(np.fromfile(str(EVAL_IMG_PATH3), dtype=np.uint8), cv2.IMREAD_COLOR)
    img4 = cv2.imdecode(np.fromfile(str(EVAL_IMG_PATH4), dtype=np.uint8), cv2.IMREAD_COLOR)
    img0 = cv2.resize(img0, (960, 540))
    img1 = cv2.resize(img1, (960, 540))
    img2 = cv2.resize(img2, (960, 540))
    img3 = cv2.resize(img3, (960, 540))
    img4 = cv2.resize(img4, (960, 540))
    return img0, img1, img2, img3, img4


def load_eval_image() -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(EVAL_IMG_PATH), dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (960, 540))
    return img


def calculate_image_similarity(image1: np.ndarray, image2: np.ndarray, similarity: float = 0.8) -> bool:
    """
    calculate image similarity, check VFI is correct

    :param image1: original image
    :param image2: upscale image
    :param similarity: similarity threshold
    :return:
    """
    # Resize the two images to the same size
    height, width = image1.shape[:2]
    image2 = cv2.resize(image2, (width, height))
    # Convert the images to grayscale
    grayscale_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayscale_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Calculate the Structural Similarity Index (SSIM) between the two images
    (score, diff) = structural_similarity(grayscale_image1, grayscale_image2, full=True)
    print("SSIM: {}".format(score))
    return score > similarity


def compare_image_size(image1: np.ndarray, image2: np.ndarray, scale: int) -> bool:
    """
    compare original image size and upscale image size, check targetscale is correct

    :param image1: original image
    :param image2: upscale image
    :param scale: upscale ratio
    :return:
    """
    target_size = (math.ceil(image1.shape[0] * scale), math.ceil(image1.shape[1] * scale))

    return image2.shape[0] == target_size[0] and image2.shape[1] == target_size[1]
