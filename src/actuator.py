import ctypes
import threading
import time
from ctypes import wintypes
from typing import Optional

from .config import AUTO_FIRE_INTERVAL

# Win32 Virtual Key Codes
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_SPACE = 0x20
VK_A = 0x41
VK_D = 0x44

# Hardware Scan Codes (PS/2 Set 1 / DirectInput)
SCAN_LEFT = 0x4B
SCAN_RIGHT = 0x4D
SCAN_SPACE = 0x39
SCAN_A = 0x1E
SCAN_D = 0x20

# Win32 Event Flags
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]

class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]

class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUT_UNION)]

_desktop_attached = False

def ensure_desktop_access():
    """
    Ensures thread is attached to the interactive user desktop ('Default').
    Caches attachment status to prevent thread thrashing and handle exhaustion.
    """
    global _desktop_attached
    if _desktop_attached:
        return True

    # 1. Set DPI awareness
    if hasattr(user32, "SetProcessDpiAwarenessContext"):
        try:
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except Exception:
            pass
    elif hasattr(user32, "SetProcessDPIAware"):
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass

    # 2. Attach desktop
    DESKTOP_ALL = 0x01FF
    hdesk = user32.OpenDesktopW("Default", 0, False, DESKTOP_ALL)
    if not hdesk:
        hdesk = user32.OpenInputDesktop(0, False, DESKTOP_ALL)
    if hdesk:
        try:
            if user32.SetThreadDesktop(hdesk):
                _desktop_attached = True
                return True
        finally:
            user32.CloseDesktop(hdesk)
    return False

def _send_key(vk: int, scan: int = 0, is_extended: bool = False, key_up: bool = False) -> int:
    """
    Sends hardware-level keyboard events compatible with both Win32 message loops
    and DirectX DirectInput.
    Specifies BOTH wVk and wScan (mapped via MapVirtualKeyW).
    """
    ensure_desktop_access()

    if not scan:
        scan = user32.MapVirtualKeyW(vk, 0)

    flags = 0
    if is_extended:
        flags |= KEYEVENTF_EXTENDEDKEY
    if key_up:
        flags |= KEYEVENTF_KEYUP

    inp = INPUT()
    inp.type = 1  # INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(
        wVk=vk,
        wScan=scan,
        dwFlags=flags,
        time=0,
        dwExtraInfo=0,
    )
    return user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

class PureKeyboardActuator:
    """
    Robust Keyboard Actuator for Chicken Invaders.
    Supports:
      1. Discrete Pulse / Bash mode (ideal for discrete RL steps: down for dt, then up)
      2. Pulsed Hold mode (ideal for continuous autopilot: maintains repeat beats while holding)
      3. Both Arrow keys and WASD keysets
    """
    def __init__(self, keyset: str = "arrows"):
        self.keyset = keyset.lower()
        if self.keyset == "wasd":
            self.vk_left, self.scan_left, self.ext_left = VK_A, SCAN_A, False
            self.vk_right, self.scan_right, self.ext_right = VK_D, SCAN_D, False
        else:
            self.vk_left, self.scan_left, self.ext_left = VK_LEFT, SCAN_LEFT, True
            self.vk_right, self.scan_right, self.ext_right = VK_RIGHT, SCAN_RIGHT, True

        self.left_pressed = False
        self.right_pressed = False
        self.space_pressed = False
        self._auto_fire_active = False
        self._auto_fire_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def press_left(self):
        with self._lock:
            if self.right_pressed:
                _send_key(self.vk_right, self.scan_right, is_extended=self.ext_right, key_up=True)
                self.right_pressed = False
            _send_key(self.vk_left, self.scan_left, is_extended=self.ext_left, key_up=False)
            self.left_pressed = True

    def press_right(self):
        with self._lock:
            if self.left_pressed:
                _send_key(self.vk_left, self.scan_left, is_extended=self.ext_left, key_up=True)
                self.left_pressed = False
            _send_key(self.vk_right, self.scan_right, is_extended=self.ext_right, key_up=False)
            self.right_pressed = True

    def release_movement(self):
        with self._lock:
            if self.left_pressed:
                _send_key(self.vk_left, self.scan_left, is_extended=self.ext_left, key_up=True)
                self.left_pressed = False
            if self.right_pressed:
                _send_key(self.vk_right, self.scan_right, is_extended=self.ext_right, key_up=True)
                self.right_pressed = False

    def pulse_left(self, duration: float = 0.04):
        """Executes a discrete bash/pulse to the left."""
        self.press_left()
        time.sleep(duration)
        self.release_movement()

    def pulse_right(self, duration: float = 0.04):
        """Executes a discrete bash/pulse to the right."""
        self.press_right()
        time.sleep(duration)
        self.release_movement()

    def step_movement(self, action: int, duration: float = 0.04):
        """
        Executes a discrete calibrated movement step for RL environments.
        0: Stay, 1: Left pulse, 2: Right pulse.
        """
        if action == 1:
            self.pulse_left(duration=duration)
        elif action == 2:
            self.pulse_right(duration=duration)
        else:
            self.release_movement()
            time.sleep(duration)

    def set_movement(self, action: int, repeat: bool = True):
        """
        Sets movement state for continuous game loops (e.g. Autopilot Engine).
        If repeat is True, re-sends keydown on existing direction to emulate hardware keyboard repeat.
        """
        if action == 1:
            self.press_left()
        elif action == 2:
            self.press_right()
        else:
            self.release_movement()

    def fire_down(self):
        with self._lock:
            _send_key(VK_SPACE, SCAN_SPACE, is_extended=False, key_up=False)
            self.space_pressed = True

    def fire_up(self):
        with self._lock:
            if self.space_pressed:
                _send_key(VK_SPACE, SCAN_SPACE, is_extended=False, key_up=True)
                self.space_pressed = False

    def fire_pulse(self, duration: float = 0.05):
        """Executes a single laser fire pulse."""
        self.fire_down()
        time.sleep(duration)
        self.fire_up()

    def start_auto_fire(self):
        if self._auto_fire_active:
            return
        self._auto_fire_active = True

        def _fire_loop():
            ensure_desktop_access()
            press_duration = 0.06
            release_duration = max(0.01, AUTO_FIRE_INTERVAL - press_duration)
            while self._auto_fire_active:
                self.fire_down()
                time.sleep(press_duration)
                self.fire_up()
                time.sleep(release_duration)

        self._auto_fire_thread = threading.Thread(target=_fire_loop, daemon=True)
        self._auto_fire_thread.start()

    def stop_auto_fire(self):
        self._auto_fire_active = False
        if self._auto_fire_thread and self._auto_fire_thread.is_alive():
            self._auto_fire_thread.join(timeout=0.2)
        self.fire_up()

    def release_all(self):
        self.stop_auto_fire()
        self.release_movement()

KeyboardActuator = PureKeyboardActuator
