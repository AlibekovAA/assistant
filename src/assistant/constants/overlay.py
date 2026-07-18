from enum import IntEnum, IntFlag
from pathlib import Path
from typing import Final

OVERLAY_ASSETS_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "overlay" / "assets"
OVERLAY_LIVE2D_MODEL: Final[str] = "live2d/Hiyori/Hiyori.model3.json"
OVERLAY_WIDTH: Final[int] = 420
OVERLAY_HEIGHT: Final[int] = 720
OVERLAY_MODEL_SCALE: Final[float] = 0.78
OVERLAY_MODEL_OFFSET_Y: Final[float] = -0.08
OVERLAY_MARGIN_X: Final[int] = 16
OVERLAY_MARGIN_Y: Final[int] = 8
OVERLAY_ACTIVE_FPS: Final[int] = 30
OVERLAY_IDLE_FPS: Final[int] = 10
OVERLAY_READY_TIMEOUT_SECONDS: Final[float] = 8.0
OVERLAY_STOP_TIMEOUT_SECONDS: Final[float] = 3.0
OVERLAY_LEVEL_SMOOTH: Final[float] = 0.5
OVERLAY_MOUTH_LERP: Final[float] = 0.5
OVERLAY_MOUTH_CLOSE_LEVEL: Final[float] = 0.006
OVERLAY_MOUTH_OPEN_LEVEL: Final[float] = 0.03
OVERLAY_WIN_CLASS: Final[str] = "MinaLive2DOverlay"


class OverlayWindowLong(IntEnum):
    GWL_EXSTYLE = -20


class OverlayWindowStyle(IntFlag):
    POPUP = 0x80000000


class OverlayWindowExStyle(IntFlag):
    LAYERED = 0x00080000
    TRANSPARENT = 0x00000020
    TOOLWINDOW = 0x00000080
    TOPMOST = 0x00000008
    NOACTIVATE = 0x08000000


class OverlayShowWindow(IntEnum):
    HIDE = 0
    SHOWNA = 8


class OverlaySetWindowPos(IntFlag):
    NOSIZE = 0x0001
    NOMOVE = 0x0002
    NOACTIVATE = 0x0010
    SHOWWINDOW = 0x0040
    HIDEWINDOW = 0x0080


class OverlayHwnd(IntEnum):
    TOPMOST = -1


class OverlayUpdateLayered(IntFlag):
    ALPHA = 0x00000002


class OverlayPeekMessage(IntFlag):
    REMOVE = 0x0001


__all__ = (
    "OVERLAY_ACTIVE_FPS",
    "OVERLAY_ASSETS_DIR",
    "OVERLAY_HEIGHT",
    "OVERLAY_IDLE_FPS",
    "OVERLAY_LEVEL_SMOOTH",
    "OVERLAY_LIVE2D_MODEL",
    "OVERLAY_MARGIN_X",
    "OVERLAY_MARGIN_Y",
    "OVERLAY_MODEL_OFFSET_Y",
    "OVERLAY_MODEL_SCALE",
    "OVERLAY_MOUTH_CLOSE_LEVEL",
    "OVERLAY_MOUTH_LERP",
    "OVERLAY_MOUTH_OPEN_LEVEL",
    "OVERLAY_READY_TIMEOUT_SECONDS",
    "OVERLAY_STOP_TIMEOUT_SECONDS",
    "OVERLAY_WIDTH",
    "OVERLAY_WIN_CLASS",
    "OverlayHwnd",
    "OverlayPeekMessage",
    "OverlaySetWindowPos",
    "OverlayShowWindow",
    "OverlayUpdateLayered",
    "OverlayWindowExStyle",
    "OverlayWindowLong",
    "OverlayWindowStyle",
)
