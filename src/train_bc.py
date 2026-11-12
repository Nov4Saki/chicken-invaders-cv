import hashlib
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import DEMOS_DIR, MODELS_DIR, POLICY_MODEL_PATH
from .policy import ChickenPilotPolicy


def generate_synthetic_demos(num_samples: int = 2500) -> Path:
    """
    Generates synthetic demonstrations matching the 40-dimensional state representation.
    Covers egg dodging (danger_center, flank clearance, escape_dir), wall bounds,
    and minion alignment.
    """
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    states = []
    actions = []

    for _ in range(num_samples):
        # Ship position
        ship_x = float(np.random.uniform(0.12, 0.88))
        ship_y = float(np.random.uniform(0.85, 0.90))
        dist_left = max(0.0, ship_x - 0.08)
        dist_right = max(0.0, 0.92 - ship_x)

        # Minion target
        target_rx = float(np.random.uniform(-0.35, 0.35))
        target_ry = float(np.random.uniform(-0.65, -0.30))
        num_enemies = float(np.random.uniform(0.2, 1.0))

        danger_center = 0.0
        danger_left = 0.0
        danger_right = 0.0
        closest_egg_rx = 0.0
        closest_egg_ry = -1.0
        escape_dir = 0.0

        egg_slots = []
        powerup_slot = [0.0, 0.0, 0.0]
        meat_slot = [0.0, 0.0, 0.0]
        scenario_roll = np.random.rand()

        if scenario_roll < 0.35:
            # 1. EVADE: Imminent threat directly overhead (|rx| < 0.055)
            rel_x = float(np.random.uniform(-0.045, 0.045))
            rel_y = float(np.random.uniform(-0.35, -0.06))
            urgency = min(5.0, 1.0 / (abs(rel_y) + 0.03))
            danger_center = urgency
            closest_egg_rx = rel_x
            closest_egg_ry = rel_y
            egg_slots.append([rel_x, rel_y, 1.0])

            # Escape direction
            if dist_left < 0.06:
                escape_dir = 1.0
                chosen_action = 2  # RIGHT
            elif dist_right < 0.06:
                escape_dir = -1.0
                chosen_action = 1  # LEFT
            elif rel_x >= 0:
                escape_dir = -1.0
                chosen_action = 1  # Move away to LEFT
            else:
                escape_dir = 1.0
                chosen_action = 2  # Move away to RIGHT

        elif scenario_roll < 0.55:
            # 2. EVADE: Flank threat (egg or meteor falling on left or right side)
            is_left_threat = np.random.rand() < 0.5
            if is_left_threat:
                rel_x = float(np.random.uniform(-0.13, -0.05))
                rel_y = float(np.random.uniform(-0.35, -0.06))
                danger_left = min(5.0, 1.0 / (abs(rel_y) + 0.03))
                if target_rx > 0.03 and dist_right > 0.04:
                    chosen_action = 2  # RIGHT
                else:
                    chosen_action = 0  # STAY
            else:
                rel_x = float(np.random.uniform(0.05, 0.13))
                rel_y = float(np.random.uniform(-0.35, -0.06))
                danger_right = min(5.0, 1.0 / (abs(rel_y) + 0.03))
                if target_rx < -0.03 and dist_left > 0.04:
                    chosen_action = 1  # LEFT
                else:
                    chosen_action = 0  # STAY

            closest_egg_rx = rel_x
            closest_egg_ry = rel_y
            egg_slots.append([rel_x, rel_y, 1.0])

        elif scenario_roll < 0.75:
            # 3. HIGH-PRIORITY SEEK: Weapon power-up falling nearby!
            powerup_rx = float(np.random.uniform(-0.35, 0.35))
            powerup_ry = float(np.random.uniform(-0.55, -0.05))
            powerup_slot = [powerup_rx, powerup_ry, 1.0]

            if abs(powerup_rx) < 0.03:
                chosen_action = 0  # STAY
            elif powerup_rx < 0 and dist_left > 0.04:
                chosen_action = 1  # LEFT
            elif powerup_rx > 0 and dist_right > 0.04:
                chosen_action = 2  # RIGHT
            else:
                chosen_action = 0

        elif scenario_roll < 0.85:
            # 4. LOWER-PRIORITY SEEK: Chicken meat / food falling nearby!
            meat_rx = float(np.random.uniform(-0.35, 0.35))
            meat_ry = float(np.random.uniform(-0.55, -0.05))
            meat_slot = [meat_rx, meat_ry, 1.0]

            if abs(meat_rx) < 0.03:
                chosen_action = 0  # STAY
            elif meat_rx < 0 and dist_left > 0.04:
                chosen_action = 1  # LEFT
            elif meat_rx > 0 and dist_right > 0.04:
                chosen_action = 2  # RIGHT
            else:
                chosen_action = 0

        else:
            # 5. KILL: Clear skies and no immediate collectibles; track and shoot minions/bosses
            if abs(target_rx) < 0.03:
                chosen_action = 0  # STAY
            elif target_rx < 0 and dist_left > 0.04:
                chosen_action = 1  # LEFT
            elif target_rx > 0 and dist_right > 0.04:
                chosen_action = 2  # RIGHT
            else:
                chosen_action = 0  # Wall constraint

        # Fill 4 threat slots (each: rx, ry, active)
        while len(egg_slots) < 4:
            if np.random.rand() < 0.3:
                ex = float(np.random.uniform(-0.4, 0.4))
                ey = float(np.random.uniform(-0.7, -0.1))
                egg_slots.append([ex, ey, 1.0])
            else:
                egg_slots.append([0.0, 0.0, 0.0])

        # Fill 3 enemy slots (each: rx, ry, is_boss, active)
        enemy_slots = [
            [target_rx, target_ry, 0.0, 1.0],
            [float(np.random.uniform(-0.4, 0.4)), float(np.random.uniform(-0.6, -0.3)), 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]

        # Construct flat 40-dim vector
        flat_state = [
            ship_x, dist_left, dist_right,
            danger_center, danger_left, danger_right, escape_dir,
            target_rx, target_ry, num_enemies,
        ]
        flat_state.extend(powerup_slot)
        flat_state.extend(meat_slot)
        for es in egg_slots[:4]:
            flat_state.extend(es)
        for en in enemy_slots[:3]:
            flat_state.extend(en)

        assert len(flat_state) == 40, f"Expected 40 dims, got {len(flat_state)}"
        states.append(flat_state)
        actions.append(chosen_action)

    synth_path = DEMOS_DIR / "stage1_synthetic_demo.npz"
    np.savez_compressed(
        synth_path,
        states=np.array(states, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
    )
    print(f"[Trainer] Generated {num_samples} synthetic training samples (40 dims) to {synth_path}")
    return synth_path

def train_behavioral_cloning(
    demo_file: Optional[Path] = None,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    states = None
    actions = None

    if demo_file is not None and demo_file.exists():
        data = np.load(demo_file)
        states = data["states"]
        actions = data["actions"]
        print(f"[Trainer] Loaded specified dataset: {demo_file}")
    else:
        # Check for all recorded human demonstrations in DEMOS_DIR
        human_demos = [
            p for p in DEMOS_DIR.glob("*.npz")
            if "synthetic" not in p.name.lower()
        ]
        if human_demos:
            all_states = []
            all_actions = []
            seen_fingerprints = set()
            for dpath in sorted(human_demos):
                try:
                    d = np.load(dpath)
                    if "states" in d and d["states"].ndim == 2 and d["states"].shape[1] == 40:
                        s_arr = d["states"]
                        # Robust deduplication by content hash
                        fp = hashlib.sha256(s_arr.tobytes()).hexdigest()
                        if fp in seen_fingerprints:
                            print(f"[Trainer] Skipped duplicate recording: {dpath.name}")
                            continue
                        seen_fingerprints.add(fp)
                        all_states.append(s_arr)
                        all_actions.append(d["actions"])
                        print(f"[Trainer] Loaded {len(s_arr)} human frames from: {dpath.name}")
                except Exception as e:
                    print(f"[Trainer] Skipped {dpath.name}: {e}")
            if all_states:
                states = np.concatenate(all_states, axis=0)
                actions = np.concatenate(all_actions, axis=0)
                print(f"[Trainer] Merged {len(states)} total frames from {len(all_states)} unique human session(s).")


    if states is None:
        print("[Trainer] No human recording found. Generating 40-dim Tripartite synthetic dataset...")
        demo_file = generate_synthetic_demos(num_samples=3500)
        data = np.load(demo_file)
        states = data["states"]
        actions = data["actions"]

    print(f"[Trainer] Total samples: {len(states)}, State shape: {states.shape}, Actions: {actions.shape}")

    # Dataset split
    num_samples = len(states)
    indices = np.random.permutation(num_samples)
    split_idx = int(0.85 * num_samples)

    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    x_train, y_train = torch.FloatTensor(states[train_idx]), torch.LongTensor(actions[train_idx])
    x_val, y_val = torch.FloatTensor(states[val_idx]), torch.LongTensor(actions[val_idx])

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trainer] Training on device: {device}")

    policy = ChickenPilotPolicy(state_dim=states.shape[1], action_dim=3).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        policy.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _ = policy(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(by)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == by).sum().item()
            total += len(by)

        scheduler.step()
        train_acc = correct / max(1, total)

        # Validation
        policy.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits, _ = policy(bx)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == by).sum().item()
                val_total += len(by)

        val_acc = val_correct / max(1, val_total)

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss/total:.4f} | "
                f"Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%"
            )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(policy.state_dict(), POLICY_MODEL_PATH)

    print(f"\n[Trainer] Training complete! Best validation accuracy: {best_val_acc*100:.1f}%")
    print(f"[Trainer] Policy saved to: {POLICY_MODEL_PATH}")

if __name__ == "__main__":
    train_behavioral_cloning()
