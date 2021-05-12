from typing import List, NamedTuple, Tuple

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
        self.memory: List[Transition] = []
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


class PrioritizedReplayMemory(Memory):
    def __init__(self, capacity: int, batch_size: int, alpha: float = 0.6,
                 start_beta: float = 0.4, annealing_duration: int = 10_000) -> None:
        super().__init__(batch_size)
        self.capacity = capacity

        self.alpha = alpha
        self.start_beta = start_beta
        self.annealing_duration = annealing_duration

        self.beta = start_beta
        self.beta_offset = (1.0 - self.start_beta) / self.annealing_duration

        self.memory: List[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)

        self.pos = 0
        self.eps = 1e-5

    def append(self, transition: Transition) -> None:
        # calculate priorization
        self.priorities[self.pos] = np.max(self.priorities) if len(self) > 0 else 1.0

        # add transition to memory
        if len(self) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.pos] = transition

        self.pos = (self.pos + 1) % self.capacity

    def _get_distribution(self) -> np.ndarray:
        priorities = self.priorities[:len(self)]

        probs = priorities**self.alpha
        probs /= probs.sum()
        return probs

    def sample(self, replace: bool = False) -> Tuple[List[Transition], List[float], List[int]]:
        probs = self._get_distribution()

        indices = np.random.choice(np.arange(len(self)), size=self.batch_size, replace=replace, p=probs)

        self.beta = min(1.0, self.beta + self.beta_offset)

        weights = (len(self) * probs[indices])**(-self.beta)
        weights /= np.max(weights)

        return [self.memory[i] for i in indices], weights, indices

    def update_priorities(self, indices, losses):
        losses = np.abs(losses) + self.eps
        self.priorities[indices] = losses
        # for idx, prio in zip(indices, losses):
        #    self.priorities[idx] = prio

    def __len__(self) -> int:
        return len(self.memory)
