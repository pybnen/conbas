from typing import Callable
import torch
from torch.distributions.categorical import Categorical


class Policy:
    def select_command(self, *kwargs):
        raise NotImplementedError


class EpsGreedyQPolicy(Policy):
    def __init__(self, eps: float, device: torch.device) -> None:
        self.eps = eps
        self.device = device

    def select_command(self, q_values: torch.Tensor) -> torch.Tensor:
        batch_size, n_commands = q_values.size()

        rand_num = torch.rand((batch_size, ))
        less_than_eps = (rand_num < self.eps).to(dtype=torch.int64, device=self.device)

        _, argmax_q_values = q_values.max(dim=1)
        rand_command_indices = torch.randint(0, n_commands, (batch_size, )).to(self.device)

        # not sure if this is faster than torch.where
        command_indices = less_than_eps * rand_command_indices + \
            (1 - less_than_eps) * argmax_q_values
        return command_indices


class AnnealedEpsGreedyQPolicy(EpsGreedyQPolicy):
    def __init__(self, start_eps: float, device: torch.device,
                 anneal_fn: Callable[[int], float]) -> None:
        super().__init__(eps=start_eps, device=device)
        self.anneal_fn = anneal_fn

    def update(self, step: int):
        self.eps = self.anneal_fn(step)


class SoftmaxPolicy(Policy):
    def select_command(self, q_values: torch.Tensor) -> torch.Tensor:
        distribution = Categorical(logits=q_values.detach().cpu())
        command_indices = distribution.sample()
        return command_indices
