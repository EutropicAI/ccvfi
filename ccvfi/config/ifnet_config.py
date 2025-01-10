from typing import Union

from ccvfi.config import CONFIG_REGISTRY
from ccvfi.type import ArchType, BaseConfig, ConfigType, ModelType


class IFNetConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.IFNET
    model: Union[ModelType, str] = ModelType.IFNet


IFNetConfigs = [
    IFNetConfig(
        name=ConfigType.IFNet_v426_heavy,
        url="https://github.com/routineLife1/ccvfi/releases/download/weights/IFNet_v426_heavy.pkl",
        hash="4cc518e172156ad6207b9c7a43364f518832d83a4325d484240493a9e2980537",
        in_frame_count=2,
    )
]

for cfg in IFNetConfigs:
    CONFIG_REGISTRY.register(cfg)
