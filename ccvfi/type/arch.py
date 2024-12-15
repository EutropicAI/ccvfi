from enum import Enum


# Enum for the architecture type, use capital letters
class ArchType(str, Enum):
    # ------------------------------------- Video Frame Interpolation ----------------------------------------------

    IFNet = "IFNet"
    DRBA = "DRBA"
