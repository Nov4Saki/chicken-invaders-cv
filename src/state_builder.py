from typing import Dict, List, Optional, Tuple
import numpy as np

class StateBuilder:
    """
    Constructs high-signal state vectors for Chicken Invaders navigation.
    Provides explicit egg-proximity radar signals (center threat, left flank, right flank,
    and safe escape clearance) so neural networks can learn reactive dodging easily.
    """
    def __init__(self, max_eggs: int = 4, max_enemies: int = 3):
        self.max_eggs = max_eggs
        self.max_enemies = max_enemies
        self.last_ship_pos: Optional[Tuple[float, float]] = None
        self.last_time: Optional[float] = None

    @property
    def state_dim(self) -> int:
        # Ship & walls: 3 (ship_x, dist_left, dist_right)
        # Radar sensors: 4 (danger_center, danger_left, danger_right, escape_dir)
        # Tactical enemy target: 3 (target_rx, target_ry, num_enemies)
        # High-Priority Power-up: 3 (powerup_rx, powerup_ry, active)
        # Lower-Priority Meat: 3 (meat_rx, meat_ry, active)
        # Detailed threat slots: max_eggs * 3 (rel_x, rel_y, active) = 12
        # Detailed enemy slots: max_enemies * 4 (rel_x, rel_y, is_boss, active) = 12
        # Total: 3 + 4 + 3 + 3 + 3 + 12 + 12 = 40
        return 3 + 4 + 3 + 3 + 3 + (self.max_eggs * 3) + (self.max_enemies * 4)

    def extract_entities(self, detections: List[Dict]) -> Dict[str, List[Dict]]:
        entities: Dict[str, List[Dict]] = {
            "ship": [],
            "threats": [],       # EVADE: egg, meteor
            "targets": [],       # KILL: chicken_minion, chicken_boss
            "collectibles": [],  # SEEK: buff_powerup, buff_chicken_leg, metalic_object
            # Explicit split
            "powerups": [],      # High-priority weapon upgrades & metal crates
            "legs": [],          # Lower-priority chicken roast / meat
            # Legacy buckets
            "eggs": [],
            "minions": [],
            "bosses": [],
        }

        for d in detections:
            cname = d["class_name"]
            if cname == "ship":
                entities["ship"].append(d)
            elif cname in ("egg", "meteor"):
                entities["threats"].append(d)
                entities["eggs"].append(d)
            elif cname == "chicken_minion":
                entities["targets"].append(d)
                entities["minions"].append(d)
            elif cname == "chicken_boss":
                entities["targets"].append(d)
                entities["bosses"].append(d)
            elif cname in ("buff_powerup", "buff_chicken_leg", "metalic_object"):
                entities["collectibles"].append(d)
                if cname in ("buff_powerup", "metalic_object"):
                    entities["powerups"].append(d)
                elif cname == "buff_chicken_leg":
                    entities["legs"].append(d)

        return entities

    def build_state(
        self,
        detections: List[Dict],
        timestamp: float,
    ) -> Tuple[np.ndarray, Dict]:
        entities = self.extract_entities(detections)

        # 1. Ship Position
        if entities["ship"]:
            best_ship = max(entities["ship"], key=lambda x: x["confidence"])
            ship_x, ship_y = best_ship["center"]
            ship_detected = 1.0
            self.last_ship_pos = (ship_x, ship_y)
        else:
            if self.last_ship_pos:
                ship_x, ship_y = self.last_ship_pos
                ship_detected = 0.5
            else:
                ship_x, ship_y = 0.5, 0.88
                ship_detected = 0.0

        # Wall clearances
        dist_to_left_wall = max(0.0, ship_x - 0.08)
        dist_to_right_wall = max(0.0, 0.92 - ship_x)

        # 2. EVADE: Parse all falling threats (eggs + meteors) & compute threat proximity radar
        parsed_threats = []
        danger_center = 0.0
        danger_left = 0.0
        danger_right = 0.0
        closest_threat_rx = 0.0
        closest_threat_ry = -1.0
        min_threat_dist = 999.0

        for threat in entities["threats"]:
            tx, ty = threat["center"]
            rx = tx - ship_x
            ry = ty - ship_y  # negative when threat is above ship
            dist = float(np.hypot(rx, ry))
            is_meteor = 1.0 if threat["class_name"] == "meteor" else 0.0

            parsed_threats.append({
                "pos": (tx, ty),
                "rel": (rx, ry),
                "dist": dist,
                "is_meteor": is_meteor,
            })

            # Check threats that are above or at the ship and within striking distance
            if -0.55 < ry < 0.04:
                urgency = 1.0 / (abs(ry) + 0.03)

                # Center direct collision zone (slightly wider for meteors)
                corridor = 0.075 if is_meteor else 0.060
                if abs(rx) < corridor:
                    danger_center = max(danger_center, urgency)
                # Left flank threat zone (-0.15 < rx < -0.04)
                elif -0.15 < rx <= -0.04:
                    danger_left = max(danger_left, urgency)
                # Right flank threat zone (0.04 <= rx < 0.15)
                elif 0.04 <= rx < 0.15:
                    danger_right = max(danger_right, urgency)

                if dist < min_threat_dist:
                    min_threat_dist = dist
                    closest_threat_rx = rx
                    closest_threat_ry = ry

        # Safe escape direction hint: +1 for right, -1 for left
        if danger_center > 0:
            if dist_to_left_wall < 0.05:
                escape_dir = 1.0  # Must go right
            elif dist_to_right_wall < 0.05:
                escape_dir = -1.0 # Must go left
            elif danger_left < danger_right:
                escape_dir = -1.0 # Left is clearer
            else:
                escape_dir = 1.0  # Right is clearer
        else:
            escape_dir = 0.0

        # Sort threats by vertical threat (closest above ship first)
        parsed_threats.sort(key=lambda t: (t["rel"][1] > 0.04, abs(t["rel"][0]) + max(0.0, -t["rel"][1])))

        # 3. KILL: Enemies (Minions + Bosses)
        all_enemies = []
        for t in entities["targets"]:
            tx, ty = t["center"]
            is_boss = 1.0 if t["class_name"] == "chicken_boss" else 0.0
            all_enemies.append({
                "abs_pos": (tx, ty),
                "rel_x": tx - ship_x,
                "rel_y": ty - ship_y,
                "is_boss": is_boss,
            })

        # Find best enemy to target (lowest down and closest column)
        if all_enemies:
            best_target = min(all_enemies, key=lambda e: (abs(e["rel_x"]) - e["abs_pos"][1] * 0.25))
            target_rx = best_target["rel_x"]
            target_ry = best_target["rel_y"]
        else:
            target_rx = 0.0
            target_ry = -0.5

        num_enemies = min(1.0, len(all_enemies) / 20.0)

        # 4. SEEK: Separate High-Priority Power-ups and Lower-Priority Chicken Meat
        # Nearest Important Power-up (Weapon Upgrade / Missile)
        if entities["powerups"]:
            best_p = min(entities["powerups"], key=lambda c: np.hypot(c["center"][0] - ship_x, c["center"][1] - ship_y))
            powerup_rx = float(best_p["center"][0] - ship_x)
            powerup_ry = float(best_p["center"][1] - ship_y)
            powerup_active = 1.0
        else:
            powerup_rx, powerup_ry, powerup_active = 0.0, 0.0, 0.0

        # Nearest Chicken Meat (Food / Score)
        if entities["legs"]:
            best_m = min(entities["legs"], key=lambda c: np.hypot(c["center"][0] - ship_x, c["center"][1] - ship_y))
            meat_rx = float(best_m["center"][0] - ship_x)
            meat_ry = float(best_m["center"][1] - ship_y)
            meat_active = 1.0
        else:
            meat_rx, meat_ry, meat_active = 0.0, 0.0, 0.0

        # 5. Construct 40-Dimensional Feature Vector
        vector = [
            ship_x,
            dist_to_left_wall,
            dist_to_right_wall,
            # Evade Threat Radar (4 floats)
            min(5.0, danger_center),
            min(5.0, danger_left),
            min(5.0, danger_right),
            escape_dir,
            # Kill Aiming Target (3 floats)
            target_rx,
            target_ry,
            num_enemies,
            # Nearest Important Power-up (High-Priority Seek - 3 floats)
            powerup_rx,
            powerup_ry,
            powerup_active,
            # Nearest Chicken Meat (Lower-Priority Seek - 3 floats)
            meat_rx,
            meat_ry,
            meat_active,
        ]

        # Top-K Nearest Threat slots (Evade - 4 slots * 3 = 12 floats)
        for i in range(self.max_eggs):
            if i < len(parsed_threats):
                pt = parsed_threats[i]
                vector.extend([pt["rel"][0], pt["rel"][1], 1.0])
            else:
                vector.extend([0.0, 0.0, 0.0])

        # Top-K Nearest Enemy slots (Kill - 3 slots * 4 = 12 floats)
        for i in range(self.max_enemies):
            if i < len(all_enemies):
                en = all_enemies[i]
                vector.extend([en["rel_x"], en["rel_y"], en["is_boss"], 1.0])
            else:
                vector.extend([0.0, 0.0, 0.0, 0.0])

        self.last_time = timestamp

        info = {
            "ship_pos": (ship_x, ship_y),
            "ship_detected": bool(entities["ship"]),
            # Tripartite counts & collections
            "num_threats": len(entities["threats"]),
            "num_targets": len(entities["targets"]),
            "num_collectibles": len(entities["collectibles"]),
            "num_powerups": len(entities["powerups"]),
            "num_meat": len(entities["legs"]),
            "nearest_powerup": (powerup_rx, powerup_ry, powerup_active),
            "nearest_meat": (meat_rx, meat_ry, meat_active),
            "parsed_threats": parsed_threats,
            "parsed_eggs": parsed_threats,  # Backwards-compatible for SafetyDodger
            "enemies": all_enemies,
            "target_rx": target_rx,
            "danger_center": danger_center,
            "escape_dir": escape_dir,
            # Legacy fields
            "num_eggs": len(entities["eggs"]),
            "num_minions": len(entities["minions"]),
            "num_bosses": len(entities["bosses"]),
        }

        return np.array(vector, dtype=np.float32), info
