import time
import numpy as np
import torch
from ultralytics import YOLO
from src.perception import GamePerception, ensure_desktop_access
from src.state_builder import StateBuilder
from src.dodger import SafetyDodger
from src.policy import ChickenPilotPolicy
from src.actuator import PureKeyboardActuator
from src.config import YOLO_IMG_SIZE


def benchmark_models(num_frames: int = 100):
    print("=" * 70)
    print("CHICKEN INVADERS: VISION MODEL BENCHMARK COMPARISON")
    print("=" * 70)

    ensure_desktop_access()
    perception = GamePerception()
    state_builder = StateBuilder()
    dodger = SafetyDodger()
    actuator = PureKeyboardActuator()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ChickenPilotPolicy(state_dim=40, action_dim=3).to(device)
    policy.eval()

    # Capture a live sample frame
    live_frame = None
    for _ in range(5):
        live_frame = perception.capture_frame()
        if live_frame is not None:
            break

    if live_frame is None:
        print("[!] Warning: Could not capture live frame, generating 1440x1080 synthetic canvas.")
        live_frame = np.zeros((1080, 1440, 3), dtype=np.uint8)

    models_to_test = [
        ("Previous Model (PyTorch)", "models/best (5).pt"),
        ("Previous Model (TorchScript)", "models/best (5).torchscript"),
        ("NEW Nano Model (PyTorch)", "models/Nano_chickent_invaders.pt"),
        ("NEW Nano Model (TorchScript)", "models/Nano_chickent_invaders.torchscript"),
    ]

    results_table = []

    for name, path in models_to_test:
        print(f"\n--> Testing: {name} ({path})")
        m = YOLO(path)
        if device.type == "cuda" and path.endswith(".pt"):
            m.to("cuda")

        # Warmup
        for _ in range(15):
            with torch.inference_mode():
                _ = m.predict(live_frame, conf=0.28, imgsz=YOLO_IMG_SIZE, verbose=False, device="cuda")

        if device.type == "cuda":
            torch.cuda.synchronize()

        t_yolo_total = []
        t_infer = []
        t_e2e = []
        detected_counts = []

        for _ in range(num_frames):
            e2e_start = time.perf_counter()

            # 1. Capture
            frame = perception.capture_frame()
            if frame is None:
                frame = live_frame

            # 2. YOLO Predict
            if device.type == "cuda":
                torch.cuda.synchronize()
            y0 = time.perf_counter()
            with torch.inference_mode():
                res = m.predict(frame, conf=0.28, imgsz=YOLO_IMG_SIZE, verbose=False, device="cuda")[0]

            if device.type == "cuda":
                torch.cuda.synchronize()
            y1 = time.perf_counter()

            t_yolo_total.append((y1 - y0) * 1000)
            if hasattr(res, "speed"):
                t_infer.append(res.speed.get("inference", (y1 - y0) * 1000))

            # 3. Box extraction
            detections = []
            h, w, _ = frame.shape
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class_id": cls_id,
                    "class_name": "entity",
                    "confidence": conf,
                    "box_pixels": (int(x1), int(y1), int(x2), int(y2)),
                    "center": (((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h),
                })
            detected_counts.append(len(detections))

            # 4. State & Policy & Dodge
            s_vec, s_info = state_builder.build_state(detections, timestamp=time.time())
            st = torch.FloatTensor(s_vec).to(device)
            act = policy.act(st)
            dodger.evaluate_threat(s_info["ship_pos"], s_info["parsed_eggs"], desired_action=act)
            actuator.set_movement(0)

            e2e_end = time.perf_counter()
            t_e2e.append((e2e_end - e2e_start) * 1000)

        mean_yolo = np.mean(t_yolo_total)
        mean_infer = np.mean(t_infer) if t_infer else mean_yolo
        mean_e2e = np.mean(t_e2e)
        fps = 1000.0 / mean_e2e
        mean_dets = np.mean(detected_counts)

        results_table.append({
            "name": name,
            "yolo_ms": mean_yolo,
            "infer_ms": mean_infer,
            "e2e_ms": mean_e2e,
            "fps": fps,
            "avg_dets": mean_dets,
        })

    perception.close()

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE (Averaged across", num_frames, "Frames)")
    print("=" * 80)
    print(f"{'Model Architecture':<32} | {'YOLO Total':<10} | {'GPU Forward':<11} | {'End-to-End':<10} | {'Loop FPS':<8}")
    print("-" * 80)
    
    baseline_e2e = results_table[0]["e2e_ms"]
    for r in results_table:
        speedup = baseline_e2e / r["e2e_ms"]
        print(
            f"{r['name']:<32} | {r['yolo_ms']:7.2f} ms | {r['infer_ms']:8.2f} ms | "
            f"{r['e2e_ms']:7.2f} ms | {r['fps']:5.1f} FPS ({speedup:.2f}x)"
        )
    print("=" * 80)

if __name__ == "__main__":
    benchmark_models(100)
