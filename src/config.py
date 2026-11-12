from dataclasses import dataclass
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DEMOS_DIR = BASE_DIR / "demos"
YOLO_MODEL_PATH = MODELS_DIR / "Nano_chickent_invaders.pt"
YOLO_TORCHSCRIPT_PATH = MODELS_DIR / "Nano_chickent_invaders.torchscript"
POLICY_MODEL_PATH = MODELS_DIR / "policy_bc.pt"

# Game Process and Window
GAME_WINDOW_TITLE = "Chicken Invaders 1"
GAME_PROCESS_NAME = "CI1.exe"

# YOLO Detected Class Mapping
CLASS_NAMES = {
    0: "buff_chicken_leg",
    1: "buff_powerup",
    2: "chicken_boss",
    3: "chicken_minion",
    4: "egg",
    5: "feathers",
    6: "laser",
    7: "metalic_object",
    8: "meteor",
    9: "ship",
}

# Detection filtering & confidence
YOLO_CONFIDENCE = 0.28
YOLO_IMG_SIZE = 384  # Optimized 384x384 resolution for high-speed TorchScript inference


# Actions
ACTION_STAY = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
NUM_ACTIONS = 3

# Safety Dodger Parameters
DODGE_SAFETY_MARGIN_X = 0.08  # Normalized screen width corridor
DODGE_DANGER_Y = 0.85         # Normalized screen height cutoff for imminent eggs
FALLING_EGG_SPEED_THRESHOLD = 0.001

# Continuous Auto-Fire
AUTO_FIRE_ENABLED = True
AUTO_FIRE_INTERVAL = 0.20  # 5 intervals per second (200ms period)
