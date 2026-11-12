import time
from typing import List
import numpy as np
import torch
import torch.nn as nn

from .config import MODELS_DIR, POLICY_MODEL_PATH
from .environment import ChickenInvadersEnv
from .policy import ChickenPilotPolicy

class PPOTrainer:
    def __init__(
        self,
        env: ChickenInvadersEnv,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RL Trainer] Initializing PPO on device: {self.device}")

        self.policy = ChickenPilotPolicy(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
        ).to(self.device)

        # Warm start from BC checkpoint if available
        if POLICY_MODEL_PATH.exists():
            try:
                print(f"[RL Trainer] Loading behavioral cloning warm-start from {POLICY_MODEL_PATH}...")
                self.policy.load_state_dict(torch.load(POLICY_MODEL_PATH, map_location=self.device))
                print("[RL Trainer] Warm-start checkpoint loaded successfully.")
            except Exception as e:
                print(f"[RL Trainer] Could not load warm-start: {e}")

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def collect_rollout(self, steps_per_rollout: int = 128):
        states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []

        state, _ = self.env.reset()
        for _ in range(steps_per_rollout):
            state_tensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                logits, value = self.policy(state_tensor.unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_state, reward, done, _, _ = self.env.step(action.item())

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())
            log_probs.append(log_prob.item())

            state = next_state
            if done:
                state, _ = self.env.reset()

        # Compute GAE (Generalized Advantage Estimation)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            _, next_val = self.policy(state_tensor.unsqueeze(0))
            next_val = next_val.item()

        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_v = next_val if t == len(rewards) - 1 else values[t + 1]
            non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_v * non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values, dtype=np.float32)

        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.LongTensor(np.array(actions)).to(self.device),
            torch.FloatTensor(np.array(log_probs)).to(self.device),
            torch.FloatTensor(advantages).to(self.device),
            torch.FloatTensor(returns).to(self.device),
            np.sum(rewards),
        )

    def train_epoch(self, states, actions, old_log_probs, advantages, returns, ppo_epochs: int = 4):
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(ppo_epochs):
            new_log_probs, entropy, new_values = self.policy.evaluate_actions(states, actions)

            # Policy loss with PPO clipping
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(new_values, returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

    def train(self, total_iterations: int = 50, steps_per_rollout: int = 128):
        print(f"[RL Trainer] Starting PPO training for {total_iterations} iterations...")
        rl_save_path = MODELS_DIR / "policy_rl.pt"

        for it in range(1, total_iterations + 1):
            states, actions, log_probs, adv, ret, ep_reward = self.collect_rollout(steps_per_rollout)
            self.train_epoch(states, actions, log_probs, adv, ret)

            print(f"[RL Iteration {it:03d}/{total_iterations:03d}] Rollout Reward: {ep_reward:.2f}")

            if it % 10 == 0:
                torch.save(self.policy.state_dict(), rl_save_path)
                print(f"[RL Trainer] Checkpoint saved to: {rl_save_path}")

        torch.save(self.policy.state_dict(), rl_save_path)
        print(f"[RL Trainer] Training finished. Final model saved to {rl_save_path}")
        self.env.close()

def main():
    env = ChickenInvadersEnv()
    trainer = PPOTrainer(env)
    try:
        trainer.train(total_iterations=30, steps_per_rollout=100)
    finally:
        env.close()

if __name__ == "__main__":
    main()
