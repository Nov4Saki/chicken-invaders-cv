import ctypes
from ctypes import wintypes
import threading
import time
from typing import Dict, List, Optional, Tuple
import cv2
import mss
import numpy as np
import torch
from ultralytics import YOLO

from .config import (
    CLASS_NAMES,
    GAME_WINDOW_TITLE,
    YOLO_CONFIDENCE,
    YOLO_IMG_SIZE,
    YOLO_MODEL_PATH,
)

user32 = ctypes.windll.user32

DESKTOP_ALL = 0x01FF

def set_dpi_aware():
    user32 = ctypes.windll.user32
    if hasattr(user32, "SetProcessDpiAwarenessContext"):
        try:
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            return
        except Exception:
            pass
    if hasattr(user32, "SetProcessDPIAware"):
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass

def ensure_desktop_access():
    set_dpi_aware()
    user32 = ctypes.windll.user32
    # Attach thread to interactive user input desktop (required for SendInput / keybd_event)
    hdesk = user32.OpenInputDesktop(0, False, DESKTOP_ALL)
    if not hdesk:
        hdesk = user32.OpenDesktopW("Default", 0, False, DESKTOP_ALL)
    if hdesk:
        user32.SetThreadDesktop(hdesk)

class GamePerception:
    def __init__(self, model_path: Optional[str] = None):
        ensure_desktop_access()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load optimized TorchScript model on GPU if available, else PyTorch checkpoint
        from .config import YOLO_TORCHSCRIPT_PATH
        if model_path is None and self.device == "cuda" and YOLO_TORCHSCRIPT_PATH.exists():
            self.model_path = str(YOLO_TORCHSCRIPT_PATH)
            print(f"Loading high-speed TorchScript YOLO model: {self.model_path}...")
        else:
            self.model_path = model_path or str(YOLO_MODEL_PATH)
            print(f"Loading YOLO model on device: {self.device}: {self.model_path}...")
        
        self.model = YOLO(self.model_path)
        if self.device == "cuda" and not self.model_path.endswith(".torchscript"):
            self.model.to("cuda")

        self.sct = mss.mss()
        self.cam = None
        self.hwnd: Optional[int] = None
        self.window_rect: Optional[Tuple[int, int, int, int]] = None
        self.play_area_crop: Optional[Tuple[int, int, int, int]] = None

        # Asynchronous YOLO inference state
        self._async_running = False
        self._async_thread: Optional[threading.Thread] = None
        self._async_lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._pending_frame: Optional[np.ndarray] = None
        self._latest_detections: List[Dict] = []
        self._latest_det_time: float = 0.0
        self._worker_fps: float = 0.0

        self.update_window_bounds()
        self._init_camera()


    def _init_camera(self, target_fps: int = 120):
        """Initializes BetterCam DXGI Direct3D hardware capture if possible."""
        if not self.play_area_crop:
            return
        try:
            import bettercam
            ensure_desktop_access()
            if self.cam is not None:
                try:
                    self.cam.stop()
                    self.cam.release()
                except Exception:
                    pass
            x1, y1, x2, y2 = self.play_area_crop
            self.cam = bettercam.create(device_idx=0, output_idx=0, output_color="BGR")
            self.cam.start(region=(x1, y1, x2, y2), target_fps=target_fps, video_mode=True)
            print(f"[Perception] DXGI Direct3D GPU screen capture initialized @ {target_fps} FPS (region: {x1},{y1} -> {x2},{y2}, video_mode=True).")
        except Exception as e:
            print(f"[Perception] Note: DXGI Direct3D capture not active ({e}). Using optimized GDI capture.")
            self.cam = None


    def find_game_window(self) -> Optional[int]:
        """Finds Chicken Invaders HWND using win32gui or EnumWindows."""
        ensure_desktop_access()
        import win32gui
        import win32process
        import psutil

        found_hwnd = None

        def enum_cb(hwnd, _):
            nonlocal found_hwnd
            if win32gui.IsWindowVisible(hwnd):
                # 1. Match game process name first (CI1.exe)
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    pname = psutil.Process(pid).name().lower()
                    if pname in ("ci1.exe", "chickeninvaders.exe", "chicken invaders.exe"):
                        found_hwnd = hwnd
                        return False
                except Exception:
                    pass

                # 2. Match exact title only if not an IDE or browser window
                title = win32gui.GetWindowText(hwnd).strip()
                if title == "Chicken Invaders 1" or title == "Chicken Invaders":
                    found_hwnd = hwnd
                    return False
            return True

        try:
            win32gui.EnumWindows(enum_cb, None)
        except Exception:
            pass

        return found_hwnd

    def update_window_bounds(self) -> bool:
        """Finds the window and computes the active 4:3 game canvas crop."""
        self.hwnd = self.find_game_window()
        if not self.hwnd:
            return False

        rect = wintypes.RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
        left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            return False

        self.window_rect = (left, top, right, bottom)

        # Chicken Invaders 1 runs in 4:3 aspect ratio.
        # If running on 16:9 widescreen, black pillarboxes exist on sides.
        target_aspect = 4.0 / 3.0
        current_aspect = width / height

        if current_aspect > target_aspect:
            # Pillarboxed on left and right
            game_w = int(height * target_aspect)
            crop_x1 = left + (width - game_w) // 2
            crop_x2 = crop_x1 + game_w
            self.play_area_crop = (crop_x1, top, crop_x2, bottom)
        else:
            self.play_area_crop = (left, top, right, bottom)

        return True

    def is_game_foreground(self) -> bool:
        """Returns True if Chicken Invaders is the currently focused foreground window."""
        if not self.hwnd:
            return False
        import win32gui
        return win32gui.GetForegroundWindow() == self.hwnd

    def focus_game_window(self) -> bool:
        """Restores and brings the Chicken Invaders window to the foreground, bypassing Windows foreground locks."""
        ensure_desktop_access()
        if not self.hwnd:
            self.update_window_bounds()
        if not self.hwnd:
            return False

        import win32con
        import win32gui
        import win32process

        try:
            if win32gui.IsIconic(self.hwnd):
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
                time.sleep(0.05)

            kernel32 = ctypes.windll.kernel32
            current_thread = kernel32.GetCurrentThreadId()
            target_thread, _ = win32process.GetWindowThreadProcessId(self.hwnd)
            fg_hwnd = user32.GetForegroundWindow()
            fg_thread = win32process.GetWindowThreadProcessId(fg_hwnd)[0] if fg_hwnd else 0

            # Attach input threads
            if fg_thread and fg_thread != current_thread:
                user32.AttachThreadInput(current_thread, fg_thread, True)
            if target_thread and target_thread != current_thread:
                user32.AttachThreadInput(current_thread, target_thread, True)

            # Bring to topmost and switch
            user32.ShowWindow(self.hwnd, win32con.SW_SHOW)
            user32.SetWindowPos(self.hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)  # HWND_TOPMOST, SWP_NOMOVE | SWP_NOSIZE
            user32.SetWindowPos(self.hwnd, -2, 0, 0, 0, 0, 0x0001 | 0x0002)  # HWND_NOTOPMOST
            user32.SwitchToThisWindow(self.hwnd, True)
            user32.BringWindowToTop(self.hwnd)
            user32.SetForegroundWindow(self.hwnd)

            # Detach input threads
            if fg_thread and fg_thread != current_thread:
                user32.AttachThreadInput(current_thread, fg_thread, False)
            if target_thread and target_thread != current_thread:
                user32.AttachThreadInput(current_thread, target_thread, False)

            time.sleep(0.1)
            return self.is_game_foreground()
        except Exception as e:
            print(f"Error focusing window: {e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """Captures the active game canvas frame (BGR format)."""
        if not self.play_area_crop:
            if not self.update_window_bounds():
                return None
            self._init_camera()

        # Dynamic window bounds check (e.g. user toggles fullscreen or moves window)
        self._check_count = getattr(self, "_check_count", 0) + 1
        if self._check_count % 30 == 0 and self.hwnd:
            rect = wintypes.RECT()
            user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
            current_rect = (rect.left, rect.top, rect.right, rect.bottom)
            if current_rect != self.window_rect:
                self.update_window_bounds()
                self._init_camera()

        # 1. High-speed Direct3D DXGI hardware capture
        if self.cam is not None:
            try:
                frame = self.cam.get_latest_frame()
                if frame is not None:
                    return frame
            except Exception:
                pass

        # 2. Optimized zero-copy GDI fallback with raw buffer slicing
        x1, y1, x2, y2 = self.play_area_crop
        monitor = {
            "left": x1,
            "top": y1,
            "width": x2 - x1,
            "height": y2 - y1,
        }

        try:
            screenshot = self.sct.grab(monitor)
            # Direct buffer view into numpy without double copy, slice BGR
            arr = np.frombuffer(screenshot.raw, dtype=np.uint8).reshape((screenshot.height, screenshot.width, 4))
            return arr[:, :, :3]
        except Exception as e:
            print(f"Screen capture error: {e}")
            self.update_window_bounds()
            return None

    def detect_entities(self, frame: np.ndarray) -> List[Dict]:
        """
        Runs YOLO detection on frame.
        Returns a list of detected entities with normalized [0, 1] coordinates.
        """
        h, w, _ = frame.shape
        use_half = (self.device == "cuda")
        with torch.inference_mode():
            results = self.model.predict(
                frame,
                conf=YOLO_CONFIDENCE,
                imgsz=YOLO_IMG_SIZE,
                verbose=False,
                device=self.device,
                half=use_half,
            )[0]

        detections = []
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            cls_arr = boxes.cls.int().cpu().numpy()
            conf_arr = boxes.conf.float().cpu().numpy()
            xyxy_arr = boxes.xyxy.cpu().numpy()

            for i in range(len(boxes)):
                cls_id = int(cls_arr[i])
                conf = float(conf_arr[i])
                x1, y1, x2, y2 = xyxy_arr[i]

                # Normalized coordinates [0, 1] relative to the active game canvas
                norm_x1 = max(0.0, min(1.0, float(x1) / w))
                norm_y1 = max(0.0, min(1.0, float(y1) / h))
                norm_x2 = max(0.0, min(1.0, float(x2) / w))
                norm_y2 = max(0.0, min(1.0, float(y2) / h))

                cx = (norm_x1 + norm_x2) / 2.0
                cy = (norm_y1 + norm_y2) / 2.0
                bw = norm_x2 - norm_x1
                bh = norm_y2 - norm_y1

                detections.append({
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                    "confidence": conf,
                    "box_pixels": (int(x1), int(y1), int(x2), int(y2)),
                    "norm_box": (norm_x1, norm_y1, norm_x2, norm_y2),
                    "center": (cx, cy),
                    "size": (bw, bh),
                })

        return detections

    def start_async_worker(self):
        """Starts background worker thread for unblocked YOLO inference."""
        if self._async_running:
            return
        self._async_running = True
        self._new_frame_event.clear()
        self._async_thread = threading.Thread(target=self._worker_loop, daemon=True, name="YOLO-AsyncWorker")
        self._async_thread.start()
        print("[Perception] Asynchronous YOLO inference worker thread started.")

    def stop_async_worker(self):
        """Stops background worker thread cleanly."""
        if not self._async_running:
            return
        self._async_running = False
        self._new_frame_event.set()
        if self._async_thread and self._async_thread.is_alive():
            self._async_thread.join(timeout=1.0)
        self._async_thread = None

    def _worker_loop(self):
        """Background thread continuously running YOLO without blocking the main control loop."""
        ensure_desktop_access()
        fps_timer = time.time()
        det_count = 0

        while self._async_running:
            if not self._new_frame_event.wait(timeout=0.05):
                continue
            self._new_frame_event.clear()

            with self._async_lock:
                if self._pending_frame is None:
                    continue
                frame_to_process = self._pending_frame
                self._pending_frame = None

            try:
                dets = self.detect_entities(frame_to_process)
            except Exception as e:
                print(f"[Perception Worker] Detection error: {e}")
                dets = []

            now = time.time()
            det_count += 1
            with self._async_lock:
                self._latest_detections = dets
                self._latest_det_time = now
                if now - fps_timer >= 0.5:
                    self._worker_fps = det_count / (now - fps_timer)
                    det_count = 0
                    fps_timer = now

    def submit_frame(self, frame: np.ndarray):
        """
        Submits the freshest frame to the YOLO worker.
        Overwrites any unstarted pending frame to maintain zero-latency responsiveness.
        """
        with self._async_lock:
            self._pending_frame = frame
        self._new_frame_event.set()

    def get_latest_detections(self) -> Tuple[List[Dict], float, float]:
        """
        Returns:
            (latest_detections, latest_timestamp, worker_fps)
        """
        with self._async_lock:
            return list(self._latest_detections), self._latest_det_time, self._worker_fps

    def close(self):
        """Releases capture resources cleanly."""
        self.stop_async_worker()
        if self.cam is not None:
            try:
                self.cam.stop()
                self.cam.release()
            except Exception:
                pass
            self.cam = None
        if self.sct is not None:
            try:
                self.sct.close()
            except Exception:
                pass

