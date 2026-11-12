import sys
import time
from typing import Optional
import cv2
import numpy as np
from pynput import keyboard
import torch

from .actuator import KeyboardActuator
from .config import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    AUTO_FIRE_ENABLED,
    POLICY_MODEL_PATH,
)
from .dodger import SafetyDodger
from .perception import GamePerception
from .policy import ChickenPilotPolicy
from .state_builder import StateBuilder

class ChickenPilotEngine:
    def __init__(
        self,
        use_policy_model: bool = True,
        show_hud: bool = False,
        pure_model: bool = False,
        target_fps: float = 100.0,
        async_inference: bool = True,
        perception: Optional[GamePerception] = None,
    ):
        self.show_hud = show_hud
        self.pure_model = pure_model
        self.target_fps = target_fps
        self.async_inference = async_inference
        self.perception = perception or GamePerception()

        self.state_builder = StateBuilder()
        self.dodger = SafetyDodger()
        self.actuator = KeyboardActuator()

        # Interleaving and tracking state
        self.last_detections: Optional[list] = None
        self.last_det_time = 0.0
        self.last_action = ACTION_STAY

        if self.async_inference:
            self.perception.start_async_worker()

        # Autopilot state toggles
        self.autopilot_engaged = True
        self.should_exit = False
        self.last_hotkey_time = 0.0


        # Load policy model if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy: Optional[ChickenPilotPolicy] = None

        if use_policy_model and POLICY_MODEL_PATH.exists():
            try:
                print(f"[Engine] Loading policy model from: {POLICY_MODEL_PATH}")
                self.policy = ChickenPilotPolicy(state_dim=self.state_builder.state_dim, action_dim=3).to(self.device)
                self.policy.load_state_dict(torch.load(POLICY_MODEL_PATH, map_location=self.device))
                self.policy.eval()
                print("[Engine] Policy model loaded successfully.")
            except Exception as e:
                print(f"[Engine] Error loading policy model: {e}. Falling back to rule-based tactical aiming.")
                self.policy = None
        else:
            print("[Engine] No policy checkpoint loaded. Operating with heuristic tactical pilot + safety dodger.")

        if self.pure_model:
            print("[Engine] Pure Model Mode: Safety Dodger is DISABLED. Model has 100% direct control.")

        # Background keyboard listener for hotkey toggles (secondary listener)
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key):
        try:
            if key in (keyboard.Key.f10, keyboard.Key.f8, keyboard.Key.pause) or (hasattr(key, "char") and key.char in ("p", "P")):
                self.toggle_autopilot()
            elif key == keyboard.Key.f12:
                self.stop_engine()
        except Exception:
            pass

    def toggle_autopilot(self):
        now = time.time()
        if now - self.last_hotkey_time < 0.3:
            return
        self.last_hotkey_time = now
        self.autopilot_engaged = not self.autopilot_engaged
        state_str = "ACTIVE (Running)" if self.autopilot_engaged else "PAUSED (Standby)"
        print(f"\n[Engine] Autopilot is now: {state_str}")
        if not self.autopilot_engaged:
            self.actuator.release_all()

    def stop_engine(self):
        print("\n[Engine] Emergency Stop Triggered!")
        self.autopilot_engaged = False
        self.actuator.release_all()
        self.should_exit = True

    def check_hardware_hotkeys(self):
        """Direct kernel keyboard polling via GetAsyncKeyState for 100% reliable pause/stop."""
        import ctypes
        user32 = ctypes.windll.user32
        now = time.time()
        if now - self.last_hotkey_time < 0.3:
            return

        # F12 (0x7B) -> Stop
        if user32.GetAsyncKeyState(0x7B) & 0x8000:
            self.last_hotkey_time = now
            self.stop_engine()
            return

        # F10 (0x79), F8 (0x77), Pause/Break (0x13), P (0x50) -> Toggle Pause
        if (
            (user32.GetAsyncKeyState(0x79) & 0x8000)
            or (user32.GetAsyncKeyState(0x77) & 0x8000)
            or (user32.GetAsyncKeyState(0x13) & 0x8000)
            or (user32.GetAsyncKeyState(0x50) & 0x8000)
        ):
            self.toggle_autopilot()

    def compute_heuristic_targeting(self, ship_pos: tuple, enemies: list) -> int:
        """Fallback tactical aiming when policy model is not active."""
        if not enemies:
            return ACTION_STAY
        target_enemy = enemies[0]
        rel_x = target_enemy["rel_x"]
        if abs(rel_x) < 0.03:
            return ACTION_STAY
        return ACTION_LEFT if rel_x < 0 else ACTION_RIGHT

    def _extrapolate_detections(self, base_detections: list, dt: float, frame_shape: tuple, ship_action: Optional[int] = None) -> list:
        """
        Extrapolates moving entities (eggs, meteors, collectibles, ship) forward by dt seconds.
        Keeps control decisions synchronized with native monitor refresh rate without waiting for YOLO.
        """
        if not base_detections or dt < 0.001:
            return base_detections or []

        h, w = frame_shape[0], frame_shape[1]
        extrapolated = []
        action = ship_action if ship_action is not None else self.last_action

        for d in base_detections:
            d_extrap = dict(d)
            cname = d.get("class_name", "")
            cx, cy = d["center"]
            bw, bh = d["size"]

            # 1. Falling lethal threats
            if cname == "egg":
                cy = min(1.0, cy + 0.28 * dt)
            elif cname == "meteor":
                cy = min(1.0, cy + 0.35 * dt)
            # 2. Collectibles (powerups, chicken legs)
            elif cname in ("buff_powerup", "buff_chicken_leg", "metalic_object"):
                cy = min(1.0, cy + 0.20 * dt)
            # 3. Ship horizontal motion based on active movement
            elif cname == "ship":
                if action == ACTION_LEFT:
                    cx = max(0.08, cx - 0.35 * dt)
                elif action == ACTION_RIGHT:
                    cx = min(0.92, cx + 0.35 * dt)


            # Recompute bounding box for accurate HUD rendering
            norm_x1 = max(0.0, cx - bw / 2.0)
            norm_y1 = max(0.0, cy - bh / 2.0)
            norm_x2 = min(1.0, cx + bw / 2.0)
            norm_y2 = min(1.0, cy + bh / 2.0)

            d_extrap["center"] = (cx, cy)
            d_extrap["norm_box"] = (norm_x1, norm_y1, norm_x2, norm_y2)
            d_extrap["box_pixels"] = (int(norm_x1 * w), int(norm_y1 * h), int(norm_x2 * w), int(norm_y2 * h))
            extrapolated.append(d_extrap)

        return extrapolated

    def run(self):

        print("=" * 60)
        print("Autonomous Chicken Invaders Autopilot")
        print("=" * 60)
        print("Controls:")
        print("  F10 / F8 / P: Toggle Autopilot (PAUSE / RESUME)")
        print("  F12:          Emergency Stop & Exit")
        print("=" * 60)

        import ctypes
        winmm = ctypes.windll.winmm
        winmm.timeBeginPeriod(1)

        def sleep_until(target_t):
            diff = target_t - time.perf_counter()
            if diff > 0.002:
                time.sleep(diff - 0.0015)
            while time.perf_counter() < target_t:
                pass

        print("\nFocusing Chicken Invaders window...")
        self.perception.focus_game_window()
        time.sleep(0.5)
        print(">>> AUTOPILOT ACTIVE! Press 'F10', 'F8', or 'P' to Pause at any time <<<\n")

        fps_timer = time.time()
        frame_count = 0
        frame_idx = 0
        action_names = ["STAY", "LEFT", "RIGHT"]

        target_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0

        while not self.should_exit:
            frame_start = time.perf_counter()

            # Check physical hardware hotkeys every frame
            self.check_hardware_hotkeys()
            if self.should_exit:
                break

            # Handle paused standby state
            if not self.autopilot_engaged:
                self.actuator.release_all()
                sys.stdout.write("\r[PAUSED] Autopilot is on standby. Press 'F10', 'F8', or 'P' to resume...          ")
                sys.stdout.flush()
                time.sleep(0.08)
                continue

            now = time.time()
            frame = self.perception.capture_frame()

            if frame is None:
                time.sleep(0.005)
                continue

            yolo_fps = 0.0
            if self.async_inference:
                self.perception.submit_frame(frame)
                raw_dets, det_time, yolo_fps = self.perception.get_latest_detections()
                # On startup before first worker pass completes, do an initial sync detect
                if not raw_dets and self.last_detections is None:
                    raw_dets = self.perception.detect_entities(frame)
                    det_time = now

                if raw_dets:
                    self.last_detections = raw_dets
                    self.last_det_time = det_time

                dt = max(0.0, min(0.15, now - self.last_det_time))
                detections = self._extrapolate_detections(self.last_detections, dt, frame.shape)
            else:
                detections = self.perception.detect_entities(frame)
                self.last_detections = detections
                self.last_det_time = now

            frame_idx += 1

            state_vec, info = self.state_builder.build_state(detections, timestamp=now)
            ship_pos = info["ship_pos"]

            # Step 1: Query Neural Policy Model
            decision_source = "CRUISE"
            desired_action = ACTION_STAY

            if self.policy is not None:
                state_tensor = torch.FloatTensor(state_vec).to(self.device)
                desired_action = self.policy.act(state_tensor, deterministic=True)
                decision_source = "MODEL"
            elif info["enemies"]:
                target_enemy = min(
                    info["enemies"],
                    key=lambda e: (abs(e["abs_pos"][0] - ship_pos[0]) - e["abs_pos"][1] * 0.2)
                )
                target_rx = target_enemy["abs_pos"][0] - ship_pos[0]
                if abs(target_rx) > 0.035:
                    desired_action = ACTION_LEFT if target_rx < 0 else ACTION_RIGHT
                else:
                    desired_action = ACTION_STAY
                decision_source = "HEURISTIC"

            # Step 2: Safety Evaluation & Model Dodge Verification
            if self.pure_model:
                chosen_action = desired_action
                is_emergency = (info["danger_center"] > 0)
                was_overridden = False
                decision_mode = "PURE MODEL"
            else:
                chosen_action, is_emergency, was_overridden = self.dodger.evaluate_threat(
                    ship_pos, info["parsed_eggs"], desired_action=desired_action
                )
                if is_emergency:
                    if not was_overridden and decision_source == "MODEL":
                        decision_mode = "MODEL DODGE"
                    elif was_overridden:
                        decision_mode = "SAFETY OVERRIDE"
                    else:
                        decision_mode = "SAFETY DODGE"
                else:
                    decision_mode = "MODEL TARGET" if decision_source == "MODEL" else "HEURISTIC AIM"

            # Step 3: Actuate Movement & 5-per-second Auto-Fire
            if self.autopilot_engaged:
                self.actuator.set_movement(chosen_action)
                self.last_action = chosen_action
                if AUTO_FIRE_ENABLED:
                    self.actuator.start_auto_fire()
            else:
                self.actuator.release_all()
                self.last_action = ACTION_STAY

            # Step 4: Render Visual HUD (only when requested via --hud)
            if self.show_hud:
                h, w, _ = frame.shape
                for d in detections:
                    x1, y1, x2, y2 = d["box_pixels"]
                    cname = d["class_name"]
                    conf = d["confidence"]

                    if cname == "ship":
                        color = (255, 200, 0)
                    elif cname == "egg":
                        color = (0, 0, 255)
                        cv2.line(frame, (int((x1+x2)/2), y2), (int((x1+x2)/2), min(h, y2 + 60)), (0, 0, 255), 2)
                    elif "minion" in cname or "boss" in cname:
                        color = (0, 255, 0)
                    else:
                        color = (255, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{cname} {conf:.2f}",
                        (x1, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )

                # Draw Ship Safety Corridor
                sx_px = int(ship_pos[0] * w)
                corridor_w = int(self.dodger.danger_corridor_x * w)
                cv2.line(frame, (sx_px - corridor_w, 0), (sx_px - corridor_w, h), (0, 255, 255), 1)
                cv2.line(frame, (sx_px + corridor_w, 0), (sx_px + corridor_w, h), (0, 255, 255), 1)

                # HUD Status Header
                hud_color = (0, 255, 0) if self.autopilot_engaged else (0, 255, 255)
                status_str = "AUTOPILOT: ACTIVE [F10/F8/P: Pause | F12: Stop]" if self.autopilot_engaged else "AUTOPILOT: PAUSED [F10/F8/P: Resume]"
                cv2.putText(frame, status_str, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2)

                if decision_mode == "MODEL DODGE":
                    mode_color = (0, 255, 255)
                elif decision_mode == "SAFETY OVERRIDE":
                    mode_color = (0, 0, 255)
                elif decision_mode == "PURE MODEL":
                    mode_color = (255, 100, 255)
                else:
                    mode_color = (0, 255, 128)

                detail_str = (
                    f"Mode: {decision_mode:<15} | Model: {action_names[desired_action]} | "
                    f"Action: {action_names[chosen_action]} | Eggs: {info['num_eggs']} | Minions: {info['num_minions']}"
                )
                cv2.putText(frame, detail_str, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, mode_color, 1)

                cv2.imshow("Chicken Invaders Autopilot HUD", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_count += 1
            if time.time() - fps_timer >= 0.5:
                fps = frame_count / (time.time() - fps_timer)
                status = "ACTIVE" if self.autopilot_engaged else "STANDBY"
                yolo_str = f"{yolo_fps:.1f}" if (self.async_inference and yolo_fps > 0) else ("ASYNC" if self.async_inference else "SYNC")
                if not info["ship_detected"] and info["num_minions"] == 0:
                    sys.stdout.write(
                        f"\r[{status}] Main FPS: {fps:.1f} | YOLO FPS: {yolo_str} | In menu or level transition...                  "
                    )
                else:
                    sys.stdout.write(
                        f"\r[{status}] Main FPS: {fps:.1f} | YOLO FPS: {yolo_str} | Mode: {decision_mode:<15} | "
                        f"Model: {action_names[desired_action]:<5} | Exec: {action_names[chosen_action]:<5} | "
                        f"Eggs: {info['num_eggs']} | Minions: {info['num_minions']}   "
                    )
                sys.stdout.flush()
                frame_count = 0
                fps_timer = time.time()

            # Precision frame pacer targeting self.target_fps
            if self.target_fps > 0:
                elapsed = time.perf_counter() - frame_start
                if elapsed < target_interval:
                    sleep_until(frame_start + target_interval)


        # Teardown
        winmm.timeEndPeriod(1)
        self.actuator.release_all()
        if self.show_hud:
            cv2.destroyAllWindows()
        self.listener.stop()
        self.perception.close()
        print("\n[Engine] Stopped cleanly.")

def main():
    engine = ChickenPilotEngine()
    engine.run()

if __name__ == "__main__":
    main()
