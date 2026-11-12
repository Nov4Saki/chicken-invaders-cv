import time
import numpy as np
import torch
from src.perception import GamePerception, ensure_desktop_access
from src.state_builder import StateBuilder
from src.dodger import SafetyDodger
from src.policy import ChickenPilotPolicy
from src.actuator import PureKeyboardActuator
from src.engine import ChickenPilotEngine
from src.config import POLICY_MODEL_PATH

def run_profiler(num_frames: int = 120):
    print("=" * 70)
    print("CHICKEN INVADERS SYSTEM PERFORMANCE PROFILER")
    print("=" * 70)

    ensure_desktop_access()
    perception = GamePerception()
    state_builder = StateBuilder()
    dodger = SafetyDodger()
    actuator = PureKeyboardActuator()
    engine_helper = ChickenPilotEngine(use_policy_model=False, target_fps=0, async_inference=False, perception=perception)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ChickenPilotPolicy(state_dim=40, action_dim=3).to(device)
    if POLICY_MODEL_PATH.exists():
        policy.load_state_dict(torch.load(POLICY_MODEL_PATH, map_location=device))
    policy.eval()

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Ensure live or synthetic sample frame
    live_frame = perception.capture_frame()
    if live_frame is None:
        print("[!] No live CI1 window found. Using synthetic 1440x1080 canvas.")
        live_frame = np.zeros((1080, 1440, 3), dtype=np.uint8)

    # =========================================================================
    # PART 1: SYNCHRONOUS BASELINE PROFILING
    # =========================================================================
    print(f"\n[Part 1] Profiling {num_frames} Synchronous Frames (YOLO on main thread)...")
    
    t_sync_capture = []
    t_sync_yolo = []
    t_sync_state = []
    t_sync_policy = []
    t_sync_dodger = []
    t_sync_total = []

    for _ in range(num_frames):
        t0 = time.perf_counter()
        
        # 1. Capture
        s0 = time.perf_counter()
        frame = perception.capture_frame()
        if frame is None:
            frame = live_frame
        s1 = time.perf_counter()
        t_sync_capture.append((s1 - s0) * 1000)

        # 2. YOLO
        y0 = time.perf_counter()
        dets = perception.detect_entities(frame)
        y1 = time.perf_counter()
        t_sync_yolo.append((y1 - y0) * 1000)

        # 3. State
        sb0 = time.perf_counter()
        s_vec, s_info = state_builder.build_state(dets, timestamp=time.time())
        sb1 = time.perf_counter()
        t_sync_state.append((sb1 - sb0) * 1000)

        # 4. Policy
        pol0 = time.perf_counter()
        st = torch.FloatTensor(s_vec).to(device)
        act = policy.act(st, deterministic=True)
        pol1 = time.perf_counter()
        t_sync_policy.append((pol1 - pol0) * 1000)

        # 5. Dodger
        dod0 = time.perf_counter()
        chosen_action, _, _ = dodger.evaluate_threat(s_info["ship_pos"], s_info["parsed_eggs"], desired_action=act)
        dod1 = time.perf_counter()
        t_sync_dodger.append((dod1 - dod0) * 1000)

        t_end = time.perf_counter()
        t_sync_total.append((t_end - t0) * 1000)

    mean_sync_total = np.mean(t_sync_total)
    sync_fps = 1000.0 / mean_sync_total

    # =========================================================================
    # PART 2: ASYNCHRONOUS PIPELINE PROFILING (100+ FPS TARGET)
    # =========================================================================
    print(f"\n[Part 2] Profiling {num_frames} Asynchronous Frames (Worker + Extrapolation)...")
    perception.start_async_worker()
    time.sleep(0.5)  # Let worker warm up

    t_async_capture = []
    t_async_ipc = []
    t_async_extrap = []
    t_async_state = []
    t_async_policy = []
    t_async_dodger = []
    t_async_total = []

    last_dets = None
    last_det_t = time.time()

    for _ in range(num_frames):
        t0 = time.perf_counter()

        # 1. Capture
        s0 = time.perf_counter()
        frame = perception.capture_frame()
        if frame is None:
            frame = live_frame
        s1 = time.perf_counter()
        t_async_capture.append((s1 - s0) * 1000)

        # 2. Async submit & retrieve
        ipc0 = time.perf_counter()
        perception.submit_frame(frame)
        raw_dets, det_t, worker_fps = perception.get_latest_detections()
        if raw_dets:
            last_dets = raw_dets
            last_det_t = det_t
        ipc1 = time.perf_counter()
        t_async_ipc.append((ipc1 - ipc0) * 1000)

        # 3. Dynamic Extrapolation
        now = time.time()
        dt = max(0.0, min(0.15, now - last_det_t))
        ex0 = time.perf_counter()
        extrap_dets = engine_helper._extrapolate_detections(last_dets, dt, frame.shape)
        ex1 = time.perf_counter()
        t_async_extrap.append((ex1 - ex0) * 1000)

        # 4. State
        sb0 = time.perf_counter()
        s_vec, s_info = state_builder.build_state(extrap_dets, timestamp=now)
        sb1 = time.perf_counter()
        t_async_state.append((sb1 - sb0) * 1000)

        # 5. Policy
        pol0 = time.perf_counter()
        st = torch.FloatTensor(s_vec).to(device)
        act = policy.act(st, deterministic=True)
        pol1 = time.perf_counter()
        t_async_policy.append((pol1 - pol0) * 1000)

        # 6. Dodger
        dod0 = time.perf_counter()
        chosen_action, _, _ = dodger.evaluate_threat(s_info["ship_pos"], s_info["parsed_eggs"], desired_action=act)
        dod1 = time.perf_counter()
        t_async_dodger.append((dod1 - dod0) * 1000)

        t_end = time.perf_counter()
        t_async_total.append((t_end - t0) * 1000)

        # Yield to allow background worker to process frames
        time.sleep(0.002)

    perception.close()


    mean_async_total = np.mean(t_async_total)
    async_fps = 1000.0 / mean_async_total

    # =========================================================================
    # SUMMARY COMPARISON TABLE
    # =========================================================================
    print("\n" + "=" * 75)
    print(f"{'PIPELINE PERFORMANCE COMPARISON':^75}")
    print("=" * 75)
    print(f"{'Pipeline Stage':<36} | {'Sync Mean (ms)':<16} | {'Async Mean (ms)':<16}")
    print("-" * 75)
    print(f"{'Screen Capture (BetterCam DXGI)':<36} | {np.mean(t_sync_capture):13.2f} ms  | {np.mean(t_async_capture):13.2f} ms")
    print(f"{'YOLO GPU Detection':<36} | {np.mean(t_sync_yolo):13.2f} ms  | {'(Background Thread)':>16}")
    print(f"{'Async Frame IPC / Hand-off':<36} | {'N/A':>16} | {np.mean(t_async_ipc):13.2f} ms")
    print(f"{'Dynamic Trajectory Extrapolation':<36} | {'N/A':>16} | {np.mean(t_async_extrap):13.2f} ms")
    print(f"{'State Builder (40 dims)':<36} | {np.mean(t_sync_state):13.2f} ms  | {np.mean(t_async_state):13.2f} ms")
    print(f"{'Policy MLP Inference (GPU)':<36} | {np.mean(t_sync_policy):13.2f} ms  | {np.mean(t_async_policy):13.2f} ms")
    print(f"{'Safety Dodger Evaluation':<36} | {np.mean(t_sync_dodger):13.2f} ms  | {np.mean(t_async_dodger):13.2f} ms")
    print("-" * 75)
    print(f"{'Total Main Loop Latency':<36} | {mean_sync_total:13.2f} ms  | {mean_async_total:13.2f} ms")
    print(f"{'Loop Throughput / FPS':<36} | {sync_fps:11.1f} FPS   | {async_fps:11.1f} FPS")
    speedup = mean_sync_total / max(0.001, mean_async_total)
    print(f"{'Framerate Speedup':<36} | {'Baseline (1.0x)':>16} | {speedup:14.2f}x")
    print("=" * 75)

if __name__ == "__main__":
    run_profiler(120)
