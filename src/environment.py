import time
from typing import Any, Dict, Optional, Tuple
import numpy as np

from .actuator import KeyboardActuator
from .config import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    AUTO_FIRE_ENABLED,
    NUM_ACTIONS,
)
from .perception import GamePerception
from .state_builder import StateBuilder

class ChickenInvadersEnv:
    """
    Real-time environment for Chicken Invaders interaction and RL training.
    """
    def __init__(self, step_delay: float = 0.04):
        self.perception = GamePerception()
        self.state_builder = StateBuilder()
        self.actuator = KeyboardActuator()
        self.step_delay = step_delay

        self.last_minion_count = 0
        self.last_boss_count = 0
        self.last_collectible_count = 0
        self.last_collectible_positions = []
        self.last_ship_pos = (0.5, 0.88)
        self.steps_survived = 0

    @property
    def state_dim(self) -> int:
        return self.state_builder.state_dim

    @property
    def action_dim(self) -> int:
        return NUM_ACTIONS

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.actuator.release_all()
        time.sleep(0.1)
        self.steps_survived = 0

        frame = self.perception.capture_frame()
        while frame is None:
            time.sleep(0.2)
            frame = self.perception.capture_frame()

        detections = self.perception.detect_entities(frame)
        state, info = self.state_builder.build_state(detections, timestamp=time.time())

        self.last_minion_count = info["num_minions"]
        self.last_boss_count = info["num_bosses"]
        self.last_collectible_count = info.get("num_collectibles", 0)
        self.last_collectible_positions = [
            d["center"] for d in detections if d["class_name"] in ("buff_powerup", "buff_chicken_leg", "metalic_object")
        ]
        self.last_ship_pos = info["ship_pos"]

        if AUTO_FIRE_ENABLED:
            self.actuator.start_auto_fire()

        return state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.steps_survived += 1

        # Execute calibrated keyboard action pulse
        self.actuator.step_movement(action, duration=self.step_delay)

        now = time.time()
        frame = self.perception.capture_frame()
        if frame is None:
            # Game window lost
            return np.zeros(self.state_dim, dtype=np.float32), -10.0, True, False, {}

        detections = self.perception.detect_entities(frame)
        next_state, info = self.state_builder.build_state(detections, timestamp=now)
        ship_pos = info["ship_pos"]

        # Reward Calculation
        reward = 0.0

        # 1. Survival bonus
        reward += 0.05

        # 2. Minion destruction bonus
        curr_minions = info["num_minions"]
        if curr_minions < self.last_minion_count:
            kills = self.last_minion_count - curr_minions
            reward += 10.0 * kills
        self.last_minion_count = curr_minions

        # 3. Boss destruction bonus
        curr_bosses = info["num_bosses"]
        if curr_bosses < self.last_boss_count:
            kills = self.last_boss_count - curr_bosses
            reward += 50.0 * kills
        self.last_boss_count = curr_bosses

        # 4. Alignment bonus (tracking minions directly above ship)
        if info["enemies"]:
            closest_enemy = info["enemies"][0]
            rel_x = abs(closest_enemy["rel_x"])
            # Bonus when lined up within firing corridor (< 0.04 normalized width)
            if rel_x < 0.04:
                reward += 0.3
            elif rel_x < 0.10:
                reward += 0.1

        # 5. Collectible pickup bonus (Seek: powerups, chicken legs, crates)
        curr_collectibles = info.get("num_collectibles", 0)
        if curr_collectibles < self.last_collectible_count:
            for c_pos in self.last_collectible_positions:
                if abs(c_pos[0] - ship_pos[0]) < 0.08 and c_pos[1] > 0.70:
                    reward += 15.0
                    break
        self.last_collectible_count = curr_collectibles
        self.last_collectible_positions = [
            d["center"] for d in detections if d["class_name"] in ("buff_powerup", "buff_chicken_leg", "metalic_object")
        ]

        # 6. Egg & meteor threat penalty & collision detection
        terminated = False
        for threat in info["parsed_eggs"]:
            rel_x, rel_y = threat["rel"]
            # Threat just above ship in collision column
            if -0.15 < rel_y < 0.02 and abs(rel_x) < 0.05:
                # Direct hit / near miss
                reward -= 5.0
                if abs(rel_x) < 0.025 and abs(rel_y) < 0.04:
                    reward -= 50.0  # Likely hit

        # 7. Death detection: if ship was detected previously and now missing for several steps
        if not info["ship_detected"] and self.steps_survived > 10:
            reward -= 30.0
            terminated = True

        self.last_ship_pos = ship_pos
        truncated = False

        return next_state, reward, terminated, truncated, info

    def close(self):
        self.actuator.release_all()
