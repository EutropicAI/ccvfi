from ccvfi.util.registry import RegistryConfigInstance

CONFIG_REGISTRY: RegistryConfigInstance = RegistryConfigInstance("CONFIG")

from ccvfi.config.rife_config import RIFEConfig  # noqa
from ccvfi.config.drba_config import DRBAConfig  # noqa
