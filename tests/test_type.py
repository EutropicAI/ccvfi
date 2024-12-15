import pytest

from ccvfi import BaseModelInterface


def test_base_class() -> None:
    with pytest.raises(TypeError):
        BaseModelInterface()  # type: ignore
