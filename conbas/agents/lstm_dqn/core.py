from typing import List, NamedTuple
import random

import numpy as np
import torch


class Transition(NamedTuple):
    observation: List[int]
    command_index: torch.Tensor
    reward: float
    next_observation: List[int]
    done: bool


class TransitionBatch(NamedTuple):
    observation: List[List[int]]
    command_index: List[torch.Tensor]
    reward: List[float]
    next_observation: List[List[int]]
    done: List[bool]


class Memory:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def sample(self, replace: bool = False) -> List[Transition]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class ReplayMemory(Memory):
    def __init__(self, capacity: int, batch_size: int) -> None:
        super().__init__(batch_size)

        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def append(self, transition: Transition) -> None:
        if len(self) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, replace: bool = False) -> List[Transition]:
        indices = np.random.choice(
            np.arange(len(self)), size=self.batch_size, replace=replace)
        return [self.memory[i] for i in indices]

    def __len__(self) -> int:
        return len(self.memory)


# TODO impl from microsft start code, read paper and reimplement it, then use it
class PrioritizedReplayMemory:
    def __init__(self, capacity: int, priority_fraction: float = 0.0) -> None:
        self.priority_fraction = priority_fraction
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory, self.beta_memory = [], []
        self.alpha_position, self.beta_position = 0, 0

    def append(self, transition: Transition, is_prior: bool = False) -> None:
        if self.priority_fraction == 0.0:
            is_prior = False
        if is_prior:
            if len(self.alpha_memory) < self.alpha_capacity:
                self.alpha_memory.append(None)
            self.alpha_memory[self.alpha_position] = transition
            self.alpha_position = (self.alpha_position + 1) % self.alpha_capacity
        else:
            if len(self.beta_memory) < self.beta_capacity:
                self.beta_memory.append(None)
            self.beta_memory[self.beta_position] = transition
            self.beta_position = (self.beta_position + 1) % self.beta_capacity

    def sample(self, batch_size: int) -> List[Transition]:
        if self.priority_fraction == 0.0:
            from_beta = min(batch_size, len(self.beta_memory))
            res = random.sample(self.beta_memory, from_beta)
        else:
            from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
            from_beta = min(batch_size - int(self.priority_fraction * batch_size), len(self.beta_memory))
            res = random.sample(self.alpha_memory, from_alpha) + random.sample(self.beta_memory, from_beta)
        random.shuffle(res)
        return res

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)
