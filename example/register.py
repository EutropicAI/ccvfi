from typing import Any

from ccvfi import AutoConfig, AutoModel, VFIBaseModel
from ccvfi.arch import IFNet
from ccvfi.config import IFNetConfig

# define your own config name and model name
cfg_name = "TESTCONFIG.pth"
model_name = "TESTMODEL"

# this should be your own config, not IFNetConfig
# extend from ccvfi.BaseConfig then implement your own config parameters
cfg = IFNetConfig(
    name=cfg_name,
    url="https://github.com/routineLife1/ccvfi/releases/tag/weights/IFNet_v426_heavy.pkl",
    hash="4cc518e172156ad6207b9c7a43364f518832d83a4325d484240493a9e2980537",
)

AutoConfig.register(cfg)


# this should be your own model
# extend from ccvfi.VFIBaseModel then implement your own model
@AutoModel.register(name=model_name)
class TESTMODEL(VFIBaseModel):
    def load_model(self) -> Any:
        cfg: IFNetConfig = self.config
        state_dict = self.get_state_dict()

        model = IFNet()

        model.load_state_dict(self.convert(state_dict), strict=True)
        model.eval().to(self.device)
        return model

    def convert(self, param):
        return {
            k.replace("module.", ""): v
            for k, v in param.items()
            if "module." in k
        }


model: TESTMODEL = AutoModel.from_pretrained(cfg_name)
