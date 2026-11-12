# Autonomous Chicken Invaders Pilot

Autonomous playing system for Chicken Invaders using real-time YOLO object detection, demonstration recording, behavioral cloning, deterministic safety egg dodging, and reinforcement learning.

## Quick start

The suite is run through `run.py`.

### 1. Record human demonstration (Stage 1)
To teach the model how you play stage 1:
```powershell
python run.py record
```
- **F7**: Start / Pause recording.
- **Left / Right arrow keys**: Move ship.
- **F9**: Save demonstration checkpoint and exit.
- Checkpoints are saved automatically to `demos/stage1_demo.npz`.

### 2. Train behavioral cloning policy
Once you have recorded a demonstration (or using the built-in synthetic baseline):
```powershell
python run.py train --epochs 40
```
This trains a PyTorch policy network on your recorded movement decisions and saves the model to `models/policy_bc.pt`.

### 3. Run autonomous autopilot
To let the AI fly and beat waves:
```powershell
python run.py play
```
- **F10**: Engage / Disengage AI autopilot.
- **F12** or **Esc**: Emergency stop and exit.
- While autopilot is engaged, auto-fire runs continuously and the engine maneuvers between eggs to eliminate chickens.

### 4. Reinforcement learning self-training
To let the model explore and optimize across multiple stages via PPO:
```powershell
python run.py rl --iterations 30 --steps 100
```
PPO warm-starts from your trained behavioral cloning checkpoint and refines movement using survival, egg dodge, and kill rewards.

## Project structure

- `run.py`: CLI entry point (`record`, `train`, `play`, `rl`).
- `src/config.py`: Game constants, class mappings, and safety margins.
- `src/perception.py`: Low-latency screen capture (`mss`) and YOLO object detection on CUDA GPU.
- `src/state_builder.py`: Extracts normalized 53-dimensional state vectors (ship, eggs, enemies, buffs).
- `src/dodger.py`: Deterministic potential-field safety dodger for urgent egg avoidance.
- `src/actuator.py`: DirectX hardware scan-code keyboard actuation (`SendInput`) and auto-fire.
- `src/policy.py`: PyTorch Actor-Critic policy model.
- `src/recorder.py`: Synchronized demonstration recorder.
- `src/train_bc.py`: Behavioral cloning trainer with synthetic baseline support.
- `src/environment.py`: Real-time Chicken Invaders game environment wrapper.
- `src/train_rl.py`: PPO reinforcement learning trainer.
