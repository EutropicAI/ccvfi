from enum import Enum


# Enum for the architecture type, use capital letters
class ArchType(str, Enum):
    # ------------------------------------- Video Frame Interpolation ----------------------------------------------

    IFNET = "IFNET"
    DRBA = "DRBA"
