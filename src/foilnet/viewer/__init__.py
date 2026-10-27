"""The interactive viewer: sliders, the solver and the surrogate side by side."""

from foilnet.viewer.server import make_server
from foilnet.viewer.service import ViewerError, ViewerModel

__all__ = ["ViewerError", "ViewerModel", "make_server"]
