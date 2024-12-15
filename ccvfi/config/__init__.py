from ccvfi.util.registry import RegistryConfigInstance

CONFIG_REGISTRY: RegistryConfigInstance = RegistryConfigInstance("CONFIG")

from ccvfi.config.ifnet_config import IFNetConfig  # noqa
from ccvfi.config.drba_config import DRBAConfig  # noqa
