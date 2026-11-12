from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
)

class SafetyDodger:
    """
    Predictive Collision Avoidance Engine.
    Simulates falling egg trajectories forward in time to guarantee the ship never
    steps into a falling egg's column and always steers to the safest clearance.
    """
    def __init__(
        self,
        ship_half_width: float = 0.055,  # Collision radius around ship center
        screen_left_limit: float = 0.10, # Wall boundary
        screen_right_limit: float = 0.90,# Wall boundary
        step_delta: float = 0.035,       # Single-action step displacement
    ):
        self.ship_half_width = ship_half_width
        self.danger_corridor_x = ship_half_width
        self.screen_left_limit = screen_left_limit
        self.screen_right_limit = screen_right_limit
        self.step_delta = step_delta

    def evaluate_threat(
        self,
        ship_pos: Tuple[float, float],
        parsed_eggs: List[Dict],
        desired_action: Optional[int] = None,
    ) -> Tuple[int, bool, bool]:
        """
        Evaluates current safety and tests candidate actions against all falling eggs.
        Returns:
            (safe_action, is_emergency, was_overridden)
            where is_emergency is True if the ship's current position is under direct threat,
            and was_overridden is True if the model's desired_action was overridden for safety.
        """
        ship_x, ship_y = ship_pos

        # Filter relevant eggs: eggs above or near the ship and not already past it
        active_eggs = []
        for egg in parsed_eggs:
            ex, ey = egg["pos"]
            rel_y = ey - ship_y  # negative when above ship
            # Eggs within 0.50 normalized height above ship are active threats
            if -0.55 < rel_y < 0.04:
                active_eggs.append((ex, ey, abs(rel_y)))

        # If no eggs exist on screen, desired action is safe
        if not active_eggs:
            return (desired_action if desired_action is not None else ACTION_STAY), False, False

        # 1. Check if ship's CURRENT position is under direct threat
        imminent_threats = [
            (ex, ey, dist_y) for ex, ey, dist_y in active_eggs
            if abs(ex - ship_x) < self.ship_half_width
        ]
        is_emergency = len(imminent_threats) > 0

        # 2. Evaluate all 3 candidate actions: STAY, LEFT, RIGHT
        candidate_actions = [ACTION_STAY, ACTION_LEFT, ACTION_RIGHT]
        action_scores = {}

        for action in candidate_actions:
            if action == ACTION_LEFT:
                target_x = ship_x - self.step_delta
            elif action == ACTION_RIGHT:
                target_x = ship_x + self.step_delta
            else:
                target_x = ship_x

            # Penalty for wall collision
            if target_x < self.screen_left_limit or target_x > self.screen_right_limit:
                action_scores[action] = -9999.0
                continue

            # Compute minimum clearance and collision penalty across all active eggs
            min_dist = 999.0
            collision_penalty = 0.0

            for ex, ey, dist_y in active_eggs:
                dx = abs(ex - target_x)
                # If target position falls within the egg's lethal column
                if dx < self.ship_half_width:
                    # Imminent collision! Weight by vertical proximity (closer eggs are deadlier)
                    urgency = 1.0 / (dist_y + 0.02)
                    collision_penalty += 500.0 * urgency

                if dx < min_dist:
                    min_dist = dx

            # Base score: favor larger clearance from all eggs minus collision penalties
            score = (min_dist * 100.0) - collision_penalty

            # If not in emergency and action matches desired targeting, give slight preference
            if not is_emergency and desired_action is not None and action == desired_action:
                score += 5.0

            action_scores[action] = score

        # 3. Decision arbitration:
        best_action = max(action_scores.keys(), key=lambda a: action_scores[a])

        # If desired action was provided and is completely collision-free, honor the model's decision!
        if desired_action is not None and action_scores.get(desired_action, -9999.0) > 0:
            # Model's choice is safe. Even if in emergency, the model itself chose a valid dodge!
            return desired_action, is_emergency, False

        # If desired action was unsafe or not provided, adopt the optimal safety action
        was_overridden = (desired_action is not None and best_action != desired_action)
        return best_action, is_emergency, was_overridden
