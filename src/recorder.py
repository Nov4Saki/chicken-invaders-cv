import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from pynput import keyboard

from .config import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    DEMOS_DIR,
)
from .perception import GamePerception
from .state_builder import StateBuilder

class DemonstrationRecorder:
    def __init__(self, show_preview: bool = False):
        DEMOS_DIR.mkdir(parents=True, exist_ok=True)
        self.show_preview = show_preview
        self.perception = GamePerception()
        self.state_builder = StateBuilder()

        # Keyboard state tracking
        self.key_left_pressed = False
        self.key_right_pressed = False
        self.is_recording = False
        self.should_exit = False
        self.save_requested = False

        # Recorded data buffers
        self.recorded_states: List[np.ndarray] = []
        self.recorded_actions: List[int] = []
        self.recorded_timestamps: List[float] = []

        # Background keyboard listener
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.listener.start()

    def _on_press(self, key):
        try:
            if key == keyboard.Key.left or (hasattr(key, "char") and key.char in ("a", "A")):
                self.key_left_pressed = True
            elif key == keyboard.Key.right or (hasattr(key, "char") and key.char in ("d", "D")):
                self.key_right_pressed = True
            elif key in (keyboard.Key.f10, keyboard.Key.f8, keyboard.Key.f7) or (hasattr(key, "char") and key.char in ("r", "R")):
                self.toggle_recording()
            elif key in (keyboard.Key.f12, keyboard.Key.f9):
                self.save_requested = True
                self.should_exit = True
        except Exception:
            pass

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        status = "ACTIVE" if self.is_recording else "PAUSED"
        print(f"\n[Recorder] Recording is now: {status}!")


    def _on_release(self, key):
        try:
            if key == keyboard.Key.left or (hasattr(key, "char") and key.char in ("a", "A")):
                self.key_left_pressed = False
            elif key == keyboard.Key.right or (hasattr(key, "char") and key.char in ("d", "D")):
                self.key_right_pressed = False
        except Exception:
            pass

    def get_current_action(self, current_ship_x: Optional[float] = None) -> int:
        import ctypes
        user32 = ctypes.windll.user32

        # 1. Direct hardware keyboard polling (VK_LEFT = 0x25, VK_RIGHT = 0x27, A = 0x41, D = 0x44)
        is_left = bool(
            (user32.GetAsyncKeyState(0x25) & 0x8000)
            or (user32.GetAsyncKeyState(0x41) & 0x8000)
            or self.key_left_pressed
        )
        is_right = bool(
            (user32.GetAsyncKeyState(0x27) & 0x8000)
            or (user32.GetAsyncKeyState(0x44) & 0x8000)
            or self.key_right_pressed
        )

        if is_left and not is_right:
            return ACTION_LEFT
        elif is_right and not is_left:
            return ACTION_RIGHT

        # 2. Physical ship trajectory fallback (if played via mouse or keys swallowed)
        if current_ship_x is not None and hasattr(self, "_last_ship_x") and self._last_ship_x is not None:
            dx = current_ship_x - self._last_ship_x
            if dx < -0.003:
                return ACTION_LEFT
            elif dx > 0.003:
                return ACTION_RIGHT

        return ACTION_STAY

    def save_demonstration(self, filename: Optional[str] = None) -> Optional[Path]:
        if not self.recorded_states:
            print("[Recorder] No frames were recorded.")
            return None

        if filename is None:
            filename = f"stage1_demo_{int(time.time())}.npz"
        out_path = DEMOS_DIR / filename

        states_arr = np.array(self.recorded_states, dtype=np.float32)
        actions_arr = np.array(self.recorded_actions, dtype=np.int64)
        timestamps_arr = np.array(self.recorded_timestamps, dtype=np.float64)

        default_path = DEMOS_DIR / "stage1_demo.npz"

        np.savez_compressed(
            out_path,
            states=states_arr,
            actions=actions_arr,
            timestamps=timestamps_arr,
        )
        np.savez_compressed(
            default_path,
            states=states_arr,
            actions=actions_arr,
            timestamps=timestamps_arr,
        )

        left_cnt = int((actions_arr == ACTION_LEFT).sum())
        right_cnt = int((actions_arr == ACTION_RIGHT).sum())
        stay_cnt = int((actions_arr == ACTION_STAY).sum())

        print("\n" + "=" * 60)
        print(f"[Recorder] Demonstration saved successfully ({len(states_arr)} frames)")
        print(f"  Actions: Left={left_cnt} | Right={right_cnt} | Stay={stay_cnt}")
        print(f"  File 1: {out_path}")
        print(f"  File 2: {default_path}")
        print("=" * 60)
        return out_path

    def check_hardware_hotkeys(self):
        """Direct kernel polling via GetAsyncKeyState for 100% reliable hotkey detection."""
        import ctypes
        user32 = ctypes.windll.user32
        now = time.time()
        if not hasattr(self, "_last_hotkey_check"):
            self._last_hotkey_check = 0.0
        if now - self._last_hotkey_check < 0.25:
            return

        # F12 (0x7B) or F9 (0x78) -> Save and Exit
        if (user32.GetAsyncKeyState(0x7B) & 0x8000) or (user32.GetAsyncKeyState(0x78) & 0x8000):
            self._last_hotkey_check = now
            self.save_requested = True
            self.should_exit = True
            return

        # F10 (0x79), F8 (0x77), F7 (0x76), Pause (0x13), R (0x52) -> Toggle Pause/Resume
        if (
            (user32.GetAsyncKeyState(0x79) & 0x8000)
            or (user32.GetAsyncKeyState(0x77) & 0x8000)
            or (user32.GetAsyncKeyState(0x76) & 0x8000)
            or (user32.GetAsyncKeyState(0x13) & 0x8000)
            or (user32.GetAsyncKeyState(0x52) & 0x8000)
        ):
            self._last_hotkey_check = now
            self.toggle_recording()

    def run(self):
        print("=" * 60)
        print("Chicken Invaders Demonstration Recorder")
        print("=" * 60)
        print("Controls while playing:")
        print("  Move:           Left / Right arrows")
        print("  Fire:           Space")
        print("  Pause / Resume: F10 / F8 / F7 / R")
        print("  Save & Finish:  F12 or F9")
        print("=" * 60)

        print("\nAttaching and focusing Chicken Invaders window...")
        self.perception.focus_game_window()
        time.sleep(1.0)

        # 3-second countdown so the player is ready
        for count in (3, 2, 1):
            print(f"Recording starts in {count}...", end="\r", flush=True)
            time.sleep(1.0)

        self.is_recording = True
        print("\n>>> RECORDING STARTED! Play stage 1 now! <<<\n")

        fps_timer = time.time()
        frame_count = 0

        try:
            while not self.should_exit:
                self.check_hardware_hotkeys()
                if self.should_exit:
                    break

                # Guarantee only Chicken Invaders frames are captured
                if not self.perception.is_game_foreground():
                    sys.stdout.write("\r[PAUSED - Click into Chicken Invaders to record!]                  ")
                    sys.stdout.flush()
                    time.sleep(0.2)
                    continue


                now = time.time()
                frame = self.perception.capture_frame()

                if frame is None:
                    time.sleep(0.1)
                    continue

                # Detect entities
                detections = self.perception.detect_entities(frame)
                state_vec, info = self.state_builder.build_state(detections, timestamp=now)
                ship_x = info["ship_pos"][0] if info["ship_detected"] else None
                current_action = self.get_current_action(ship_x)
                if ship_x is not None:
                    self._last_ship_x = ship_x

                if self.is_recording:
                    self.recorded_states.append(state_vec)
                    self.recorded_actions.append(current_action)
                    self.recorded_timestamps.append(now)

                act_str = ["STAY", "LEFT", "RIGHT"][current_action]

                # Optional visual preview (only if user explicitly passes --preview)
                if self.show_preview:
                    h, w, _ = frame.shape
                    for d in detections:
                        x1, y1, x2, y2 = d["box_pixels"]
                        cname = d["class_name"]
                        color = (0, 0, 255) if cname == "egg" else ((0, 255, 0) if "minion" in cname else (255, 200, 0))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    status_text = f"REC | Frames: {len(self.recorded_states)} | Action: {act_str}"
                    cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Chicken Invaders Recorder Preview", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.save_requested = True
                        break


                frame_count += 1
                if time.time() - fps_timer >= 0.5:
                    fps = frame_count / (time.time() - fps_timer)
                    status = "REC" if self.is_recording else "PAUSED"
                    sys.stdout.write(
                        f"\r[{status}] Frames: {len(self.recorded_states):04d} | "
                        f"FPS: {fps:.1f} | Action: {act_str:<5} | "
                        f"Evade: {info['num_threats']} | Kill: {info['num_targets']} | Seek: {info['num_collectibles']}   "
                    )
                    sys.stdout.flush()
                    frame_count = 0
                    fps_timer = time.time()

        except KeyboardInterrupt:
            print("\n[Recorder] Stop requested (Ctrl+C). Saving demonstration...")

        finally:
            if self.show_preview:
                cv2.destroyAllWindows()

            self.listener.stop()
            self.perception.close()

            if self.recorded_states:
                self.save_demonstration()
            else:
                print("\n[Recorder] No frames logged.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="Show OpenCV visual preview window")
    args = parser.parse_args()

    recorder = DemonstrationRecorder(show_preview=args.preview)
    recorder.run()

if __name__ == "__main__":
    main()
