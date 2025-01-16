from typing import Any

from ccvfi import AutoConfig, AutoModel
from ccvfi.config import RIFEConfig
from ccvfi.model import RIFEModel


def test_auto_class_register() -> None:
    cfg_name = "TESTCONFIG.pth"
    model_name = "TESTMODEL"

    cfg = RIFEConfig(
        name=cfg_name,
        model=model_name,
        url="https://github.com/routineLife1/ccvfi/releases/download/weights/IFNet_v426_heavy.pkl",
        hash="4cc518e172156ad6207b9c7a43364f518832d83a4325d484240493a9e2980537",
        in_frame_count=2,
    )

    AutoConfig.register(cfg)

    @AutoModel.register(name=model_name)
    class TESTMODEL(RIFEModel):
        def load_model(self) -> Any:
            return None

        def get_cfg(self) -> Any:
            return self.config

    model: TESTMODEL = AutoModel.from_pretrained(cfg_name)
    assert model.get_cfg() == cfg
