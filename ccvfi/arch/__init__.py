from ccvfi.util.registry import Registry

ARCH_REGISTRY: Registry = Registry("ARCH")

from ccvfi.arch.ifnet_arch import IFNet  # noqa
from ccvfi.arch.drba_arch import DRBA  # noqa
