from typing import Union

from ccvfi.config import CONFIG_REGISTRY
from ccvfi.type import ArchType, BaseConfig, ConfigType, ModelType


class DRBAConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.DRBA
    model: Union[ModelType, str] = ModelType.DRBA


DRBAConfigs = [
    DRBAConfig(
        name=ConfigType.DRBA_IFNet,
        url="https://github.com/routineLife1/ccvfi/releases/download/weights/DRBA_IFNet.pkl",
        hash="4cc518e172156ad6207b9c7a43364f518832d83a4325d484240493a9e2980537",
    )
]

for cfg in DRBAConfigs:
    CONFIG_REGISTRY.register(cfg)
