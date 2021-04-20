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


class LinearAnnealedEpsGreedyQPolicy(Policy):
    def __init__(self, start_eps: float, end_eps: float, duration: int, device: torch.device) -> None:
        assert start_eps > end_eps
        assert duration > 0

        self.start_eps = start_eps
        self.end_eps = end_eps
        self.duration = duration
        self.delta_eps = (self.start_eps - self.end_eps) / duration

        self.policy = EpsGreedyQPolicy(self.start_eps, device)

    @property
    def eps(self) -> float:
        return self.policy.eps

    @eps.setter
    def eps(self, eps) -> None:
        self.policy.eps = eps

    def reset(self) -> None:
        self.eps = self.start_eps
        # self.eps = self.start_eps

    def select_command(self, q_values: torch.Tensor) -> torch.Tensor:
        command_indices = self.policy.select_command(q_values)
        # anneal eps
        self.eps = max(self.eps - self.delta_eps, self.end_eps)
        return command_indices


class SoftmaxPolicy(Policy):
    def select_command(self, q_values: torch.Tensor) -> torch.Tensor:
        distribution = Categorical(logits=q_values.detach().cpu())
        command_indices = distribution.sample()
        return command_indices
