# dqn.py

import torch
import torch.nn as nn
from einops import rearrange
from gymnasium import Env
import random
from typing import Dict
from collections import OrderedDict


class DQN(nn.Module):
    """Basic Deep Q network."""

    def __init__(self, input_dim: int = 3, action_space: int = 8, dim: int = 128):
        super().__init__()

        self.input_dim = input_dim
        self.action_space = action_space
        self.dim = dim

        # TODO: Replace with a more powerful encoder
        self.encoder = nn.Conv2d(input_dim, dim, 1)
        self.action_net = nn.Linear(dim, action_space)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        x = self.encoder(frame)
        x = rearrange(x, "b c ... -> b c (...)").mean(-1)
        return self.action_net(x)


@torch.no_grad
def epsilon_greedy(
    env: Env,
    model: nn.Module,
    obs: torch.Tensor,
    epsilon: float,
    device: torch.device,
    dtype: torch.dtype,
):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        obs = obs.to(device, dtype=dtype).unsqueeze(0)
        return model(obs).argmax().item()


@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.995):
    """Exponential moving average model updates."""
    ema_params = OrderedDict(ema_model.named_parameters())
    if hasattr(model, "module"):
        model_params = OrderedDict(model.module.named_parameters())
    else:
        model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
