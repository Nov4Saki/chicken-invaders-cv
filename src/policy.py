import torch
import torch.nn as nn
from typing import Tuple
from .config import NUM_ACTIONS

class ChickenPilotPolicy(nn.Module):
    """
    Unified Actor-Critic policy network.
    Supports both Behavioral Cloning (imitation learning) and Reinforcement Learning (PPO).
    """
    def __init__(self, state_dim: int = 40, action_dim: int = NUM_ACTIONS, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Actor head: maps state features to action logits (Stay, Left, Right)
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic head: estimates state value V(s) for RL
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, state_value)."""
        features = self.feature_net(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    def act(self, state: torch.Tensor, deterministic: bool = True) -> int:
        """Selects an action given a single state."""
        self.eval()
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            logits, _ = self.forward(state)
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
        return action

    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates actions for PPO updates."""
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value.squeeze(-1)
