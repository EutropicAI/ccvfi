from typing import Optional, Union

from pydantic import BaseModel, FilePath, HttpUrl

from ccvfi.type.arch import ArchType
from ccvfi.type.model import ModelType


class BaseConfig(BaseModel):
    name: str
    url: Optional[HttpUrl] = None
    path: Optional[FilePath] = None
    hash: Optional[str] = None
    arch: Union[ArchType, str]
    model: Union[ModelType, str]
    in_frame_count: int = 3
