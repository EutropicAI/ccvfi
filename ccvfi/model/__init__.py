from ccvfi.util.registry import Registry

MODEL_REGISTRY: Registry = Registry("MODEL")

from ccvfi.model.vfi_base_model import VFIBaseModel  # noqa
from ccvfi.model.rife_model import RIFEModel  # noqa
from ccvfi.model.drba_model import DRBAModel  # noqa
