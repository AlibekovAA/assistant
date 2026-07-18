import ctypes
from ctypes import wintypes
from enum import Enum, auto
import os
from pathlib import Path
import queue
import sys
import threading
from typing import Final

import live2d.v3 as live2d
from live2d.v3 import StandardParams
import numpy as np
from OpenGL.GL import GL_BACK, GL_BGRA, GL_UNSIGNED_BYTE, glFinish, glReadBuffer, glReadPixels
import pygame
from pygame.locals import DOUBLEBUF, HIDDEN, NOFRAME, OPENGL

from assistant.constants.overlay import (
    OVERLAY_ACTIVE_FPS,
    OVERLAY_ASSETS_DIR,
    OVERLAY_HEIGHT,
    OVERLAY_IDLE_FPS,
    OVERLAY_LEVEL_SMOOTH,
    OVERLAY_LIVE2D_MODEL,
    OVERLAY_MARGIN_X,
    OVERLAY_MARGIN_Y,
    OVERLAY_MODEL_OFFSET_Y,
    OVERLAY_MODEL_SCALE,
    OVERLAY_MOUTH_CLOSE_LEVEL,
    OVERLAY_MOUTH_LERP,
    OVERLAY_MOUTH_OPEN_LEVEL,
    OVERLAY_READY_TIMEOUT_SECONDS,
    OVERLAY_STOP_TIMEOUT_SECONDS,
    OVERLAY_WIDTH,
    OVERLAY_WIN_CLASS,
    OverlayHwnd,
    OverlayPeekMessage,
    OverlaySetWindowPos,
    OverlayShowWindow,
    OverlayUpdateLayered,
    OverlayWindowExStyle,
    OverlayWindowStyle,
)
from assistant.core.exceptions import OverlayError
from assistant.logger import Logger

_LOG = Logger.get(__name__)
_USER32 = ctypes.windll.user32 if sys.platform == "win32" else None
_GDI32 = ctypes.windll.gdi32 if sys.platform == "win32" else None
_KERNEL32 = ctypes.windll.kernel32 if sys.platform == "win32" else None
_AC_SRC_OVER: Final[int] = 0
_AC_SRC_ALPHA: Final[int] = 1
_DIB_RGB_COLORS: Final[int] = 0
_BI_RGB: Final[int] = 0
_LRESULT = ctypes.c_ssize_t
_WPARAM = ctypes.c_size_t
_LPARAM = ctypes.c_ssize_t
_WNDPROC = ctypes.WINFUNCTYPE(_LRESULT, wintypes.HWND, ctypes.c_uint, _WPARAM, _LPARAM)

if _USER32 is not None:
    _USER32.DefWindowProcW.argtypes = [wintypes.HWND, ctypes.c_uint, _WPARAM, _LPARAM]
    _USER32.DefWindowProcW.restype = _LRESULT
    _USER32.UpdateLayeredWindow.argtypes = [
        wintypes.HWND,
        wintypes.HDC,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.HDC,
        ctypes.c_void_p,
        wintypes.COLORREF,
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    _USER32.UpdateLayeredWindow.restype = wintypes.BOOL


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = (
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    )


class _BitmapInfo(ctypes.Structure):
    _fields_ = (
        ("bmiHeader", _BitmapInfoHeader),
        ("bmiColors", ctypes.c_uint32 * 3),
    )


class _Point(ctypes.Structure):
    _fields_ = (
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
    )


class _Size(ctypes.Structure):
    _fields_ = (
        ("cx", ctypes.c_int32),
        ("cy", ctypes.c_int32),
    )


class _BlendFunction(ctypes.Structure):
    _fields_ = (
        ("BlendOp", ctypes.c_byte),
        ("BlendFlags", ctypes.c_byte),
        ("SourceConstantAlpha", ctypes.c_byte),
        ("AlphaFormat", ctypes.c_byte),
    )


class _WndClassW(ctypes.Structure):
    _fields_ = (
        ("style", ctypes.c_uint),
        ("lpfnWndProc", _WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", ctypes.c_void_p),
        ("hCursor", ctypes.c_void_p),
        ("hbrBackground", ctypes.c_void_p),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    )


class _Command(Enum):
    SHOW = auto()
    HIDE = auto()
    SHUTDOWN = auto()


def _overlay_wnd_proc_impl(hwnd: int, message: int, wparam: int, lparam: int) -> int:
    if _USER32 is None:
        return 0
    return int(_USER32.DefWindowProcW(hwnd, message, wparam, lparam))


_overlay_wnd_proc = _WNDPROC(_overlay_wnd_proc_impl)


class _LayeredPresenter:
    def __init__(self, width: int, height: int) -> None:
        if _USER32 is None or _GDI32 is None:
            raise OverlayError("Layered overlay requires Windows")

        self._width = width
        self._height = height
        self._bits = ctypes.c_void_p()
        self._hdc_screen = _USER32.GetDC(0)
        self._hdc_mem = _GDI32.CreateCompatibleDC(self._hdc_screen)

        bitmap_info = _BitmapInfo()
        bitmap_info.bmiHeader.biSize = ctypes.sizeof(_BitmapInfoHeader)
        bitmap_info.bmiHeader.biWidth = width
        bitmap_info.bmiHeader.biHeight = height
        bitmap_info.bmiHeader.biPlanes = 1
        bitmap_info.bmiHeader.biBitCount = 32
        bitmap_info.bmiHeader.biCompression = _BI_RGB

        self._hbm = _GDI32.CreateDIBSection(
            self._hdc_mem,
            ctypes.byref(bitmap_info),
            _DIB_RGB_COLORS,
            ctypes.byref(self._bits),
            None,
            0,
        )
        if not self._hbm or not self._bits:
            self.close()
            raise OverlayError("Failed to create layered bitmap")

        self._old_obj = _GDI32.SelectObject(self._hdc_mem, self._hbm)
        self._blend = _BlendFunction(_AC_SRC_OVER, 0, 255, _AC_SRC_ALPHA)
        self._size = _Size(width, height)
        self._src = _Point(0, 0)

    def present(self, hwnd: int, pos_x: int, pos_y: int) -> None:
        if _USER32 is None or self._bits.value is None:
            return

        glReadBuffer(GL_BACK)
        raw = glReadPixels(0, 0, self._width, self._height, GL_BGRA, GL_UNSIGNED_BYTE)
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape((self._height, self._width, 4)).copy()
        alpha = pixels[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
        pixels[:, :, :3] = (pixels[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
        ctypes.memmove(self._bits, pixels.ctypes.data, int(pixels.nbytes))

        destination = _Point(pos_x, pos_y)
        ok = _USER32.UpdateLayeredWindow(
            hwnd,
            None,
            ctypes.byref(destination),
            ctypes.byref(self._size),
            self._hdc_mem,
            ctypes.byref(self._src),
            0,
            ctypes.byref(self._blend),
            int(OverlayUpdateLayered.ALPHA),
        )
        if not ok:
            _LOG.warning("UpdateLayeredWindow failed: %s", ctypes.GetLastError())

    def close(self) -> None:
        if _USER32 is None or _GDI32 is None:
            return
        if getattr(self, "_old_obj", None) is not None and getattr(self, "_hdc_mem", None):
            _GDI32.SelectObject(self._hdc_mem, self._old_obj)
            self._old_obj = None
        if getattr(self, "_hbm", None):
            _GDI32.DeleteObject(self._hbm)
            self._hbm = None
        if getattr(self, "_hdc_mem", None):
            _GDI32.DeleteDC(self._hdc_mem)
            self._hdc_mem = None
        if getattr(self, "_hdc_screen", None):
            _USER32.ReleaseDC(0, self._hdc_screen)
            self._hdc_screen = None


class Live2dAvatarOverlay:
    def __init__(self, *, model_path: Path | None = None) -> None:
        self._model_path = model_path or OVERLAY_ASSETS_DIR / OVERLAY_LIVE2D_MODEL
        self._commands: queue.SimpleQueue[_Command] = queue.SimpleQueue()
        self._level_lock = threading.Lock()
        self._latest_level: float = 0.0
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._started = False

    def initialize(self) -> None:
        if not self._model_path.is_file():
            raise OverlayError(f"Live2D model is missing: {self._model_path}")
        if sys.platform != "win32":
            raise OverlayError("Live2D overlay currently supports Windows only")
        self._ready.clear()
        self._stopped.clear()
        self._started = False

    def run(self) -> None:
        if self._started:
            return
        self._started = True
        self._ready.clear()
        self._stopped.clear()
        try:
            self._run_mainloop()
        finally:
            self._ready.clear()
            self._stopped.set()

    def wait_until_ready(self, timeout: float = OVERLAY_READY_TIMEOUT_SECONDS) -> bool:
        return self._ready.wait(timeout=timeout)

    def shutdown(self) -> None:
        if self._stopped.is_set():
            return
        self._commands.put(_Command.SHUTDOWN)
        if threading.current_thread() is threading.main_thread():
            return
        if not self._stopped.wait(timeout=OVERLAY_STOP_TIMEOUT_SECONDS):
            _LOG.warning("Avatar overlay did not stop in time")

    def show(self) -> None:
        self._reset_level()
        self._commands.put(_Command.SHOW)

    def hide(self) -> None:
        self._reset_level()
        self._commands.put(_Command.HIDE)

    def set_level(self, level: float) -> None:
        with self._level_lock:
            self._latest_level = level

    def _reset_level(self) -> None:
        with self._level_lock:
            self._latest_level = 0.0

    def _read_level(self) -> float:
        with self._level_lock:
            return self._latest_level

    def _run_mainloop(self) -> None:
        live2d.enableLog(False)
        pygame.init()
        live2d.init()

        screen_w, screen_h = _screen_size()
        pos_x = max(0, screen_w - OVERLAY_WIDTH - OVERLAY_MARGIN_X)
        pos_y = max(0, screen_h - OVERLAY_HEIGHT - OVERLAY_MARGIN_Y)

        os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
        pygame.display.gl_set_attribute(pygame.GL_ALPHA_SIZE, 8)
        pygame.display.set_mode(
            (OVERLAY_WIDTH, OVERLAY_HEIGHT),
            DOUBLEBUF | OPENGL | NOFRAME | HIDDEN,
            vsync=0,
        )
        gl_hwnd = _pygame_hwnd()
        if gl_hwnd is not None:
            _set_window_visible(gl_hwnd, False)

        live2d.glInit()
        model = live2d.LAppModel()
        model.LoadModelJson(str(self._model_path))
        model.Resize(OVERLAY_WIDTH, OVERLAY_HEIGHT)
        model.SetScale(OVERLAY_MODEL_SCALE)
        model.SetOffset(0.0, OVERLAY_MODEL_OFFSET_Y)
        model.SetAutoBlinkEnable(True)
        model.SetAutoBreathEnable(True)

        overlay_hwnd = _create_overlay_hwnd(pos_x, pos_y, OVERLAY_WIDTH, OVERLAY_HEIGHT)
        presenter = _LayeredPresenter(OVERLAY_WIDTH, OVERLAY_HEIGHT)
        _set_window_visible(overlay_hwnd, False)

        visible = False
        smooth_level = 0.0
        mouth = 0.0
        running = True
        clock = pygame.time.Clock()

        self._ready.set()
        _LOG.info("Avatar overlay ready (Live2D)")

        try:
            while running:
                _pump_win_messages()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        continue

                while True:
                    try:
                        command = self._commands.get_nowait()
                    except queue.Empty:
                        break

                    match command:
                        case _Command.SHOW:
                            self._reset_level()
                            smooth_level = 0.0
                            mouth = 0.0
                            model.SetParameterValue(StandardParams.ParamMouthOpenY, 0.0)
                            model.StartRandomMotion("Idle")
                            _set_window_visible(overlay_hwnd, True)
                            _raise_topmost(overlay_hwnd)
                            visible = True
                        case _Command.HIDE:
                            self._reset_level()
                            smooth_level = 0.0
                            mouth = 0.0
                            model.SetParameterValue(StandardParams.ParamMouthOpenY, 0.0)
                            _set_window_visible(overlay_hwnd, False)
                            visible = False
                        case _Command.SHUTDOWN:
                            running = False

                if visible:
                    smooth_level = (OVERLAY_LEVEL_SMOOTH * smooth_level) + (
                        (1.0 - OVERLAY_LEVEL_SMOOTH) * max(0.0, self._read_level())
                    )
                    target = _mouth_amount(smooth_level)
                    mouth += (target - mouth) * OVERLAY_MOUTH_LERP
                    model.Update()
                    model.SetParameterValue(StandardParams.ParamMouthOpenY, mouth)
                    live2d.clearBuffer(0.0, 0.0, 0.0, 0.0)
                    model.Draw()
                    glFinish()
                    presenter.present(overlay_hwnd, pos_x, pos_y)
                    pygame.display.flip()
                    clock.tick(OVERLAY_ACTIVE_FPS)
                else:
                    clock.tick(OVERLAY_IDLE_FPS)
        finally:
            presenter.close()
            if _USER32 is not None:
                _USER32.DestroyWindow(overlay_hwnd)
            live2d.dispose()
            pygame.quit()


def _screen_size() -> tuple[int, int]:
    if _USER32 is not None:
        return int(_USER32.GetSystemMetrics(0)), int(_USER32.GetSystemMetrics(1))
    info = pygame.display.Info()
    return int(info.current_w), int(info.current_h)


def _mouth_amount(level: float) -> float:
    if level <= OVERLAY_MOUTH_CLOSE_LEVEL:
        return 0.0
    if level >= OVERLAY_MOUTH_OPEN_LEVEL:
        return 1.0
    progress = (level - OVERLAY_MOUTH_CLOSE_LEVEL) / (OVERLAY_MOUTH_OPEN_LEVEL - OVERLAY_MOUTH_CLOSE_LEVEL)
    return progress * progress * (3.0 - (2.0 * progress))


def _pygame_hwnd() -> int | None:
    if _USER32 is None:
        return None
    info = pygame.display.get_wm_info()
    window = info.get("window")
    if window is None:
        return None
    return int(window)


def _create_overlay_hwnd(pos_x: int, pos_y: int, width: int, height: int) -> int:
    if _USER32 is None or _KERNEL32 is None:
        raise OverlayError("Layered overlay requires Windows")

    instance = _KERNEL32.GetModuleHandleW(None)
    class_name = OVERLAY_WIN_CLASS
    window_class = _WndClassW()
    window_class.style = 0
    window_class.lpfnWndProc = _overlay_wnd_proc
    window_class.cbClsExtra = 0
    window_class.cbWndExtra = 0
    window_class.hInstance = instance
    window_class.hIcon = None
    window_class.hCursor = None
    window_class.hbrBackground = None
    window_class.lpszMenuName = None
    window_class.lpszClassName = class_name

    atom = _USER32.RegisterClassW(ctypes.byref(window_class))
    if not atom and ctypes.GetLastError() not in {0, 1410}:
        raise OverlayError(f"RegisterClassW failed: {ctypes.GetLastError()}")

    ex_style = int(
        OverlayWindowExStyle.LAYERED
        | OverlayWindowExStyle.TRANSPARENT
        | OverlayWindowExStyle.TOOLWINDOW
        | OverlayWindowExStyle.TOPMOST
        | OverlayWindowExStyle.NOACTIVATE
    )
    hwnd = int(
        _USER32.CreateWindowExW(
            ex_style,
            class_name,
            "Mina",
            int(OverlayWindowStyle.POPUP),
            pos_x,
            pos_y,
            width,
            height,
            None,
            None,
            instance,
            None,
        )
    )
    if not hwnd:
        raise OverlayError(f"CreateWindowExW failed: {ctypes.GetLastError()}")
    return hwnd


def _raise_topmost(hwnd: int) -> None:
    if _USER32 is None:
        return
    _USER32.SetWindowPos(
        hwnd,
        OverlayHwnd.TOPMOST,
        0,
        0,
        0,
        0,
        int(OverlaySetWindowPos.NOMOVE | OverlaySetWindowPos.NOSIZE | OverlaySetWindowPos.NOACTIVATE),
    )


def _set_window_visible(hwnd: int, visible: bool) -> None:
    if _USER32 is None:
        return
    _USER32.ShowWindow(hwnd, OverlayShowWindow.SHOWNA if visible else OverlayShowWindow.HIDE)


def _pump_win_messages() -> None:
    if _USER32 is None:
        return
    message = wintypes.MSG()
    while _USER32.PeekMessageW(
        ctypes.byref(message),
        None,
        0,
        0,
        int(OverlayPeekMessage.REMOVE),
    ):
        _USER32.TranslateMessage(ctypes.byref(message))
        _USER32.DispatchMessageW(ctypes.byref(message))
