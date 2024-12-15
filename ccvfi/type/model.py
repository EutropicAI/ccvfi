from enum import Enum


# Enum for model type, use original name
class ModelType(str, Enum):
    # ------------------------------------- Video Frame Interpolation ----------------------------------------------

    IFNet = "IFNet"
    DRBA = "DRBA"
