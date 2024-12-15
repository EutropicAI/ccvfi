from enum import Enum


# Enum for config type, {ModelType.model}_{config_name}_{scale}x.pth
# For the Auxiliary Network, {ModelType.model}_{config_name}.pth
class ConfigType(str, Enum):
    # ------------------------------------- Video Frame Interpolation ----------------------------------------------

    # RIFE (IFNet)
    IFNet_v426_heavy = "IFNet_v426_heavy.pkl"

    # DRBA
    DRBA_IFNet = "DRBA_IFNet.pkl"
