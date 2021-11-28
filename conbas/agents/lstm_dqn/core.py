from typing import List, NamedTuple, Tuple, Callable

import numpy as np
import torch
import random


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
    def __init__(self, capacity: int, batch_size: int,
                 anneal_fn: Callable[[int], float], alpha: float = 0.6, start_beta: float = 0.4) -> None:
        super().__init__(batch_size)
        self.capacity = capacity

        self.alpha = alpha
        self.beta = start_beta
        self.anneal_fn = anneal_fn

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

        weights = (len(self) * probs[indices])**(-self.beta)
        weights /= np.max(weights)
        weights = np.array(weights, dtype=np.float32)

        return [self.memory[i] for i in indices], weights, indices

    def update_priorities(self, indices, losses):
        losses = np.abs(losses) + self.eps
        self.priorities[indices] = losses
        # for idx, prio in zip(indices, losses):
        #    self.priorities[idx] = prio

    def update_beta(self, step: int):
        self.beta = self.anneal_fn(step)

    def __len__(self) -> int:
        return len(self.memory)


class CCPrioritizedReplayMemory(Memory):

    def __init__(self, capacity: int, batch_size: int, priority_fraction=0.0):
        super().__init__(batch_size)

        # prioritized replay memory
        self.priority_fraction = priority_fraction
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory, self.beta_memory = [], []
        self.alpha_position, self.beta_position = 0, 0

        self.beta = 0

    def append(self, transition: Transition) -> None:
        """Saves a transition."""
        is_prior = transition.reward > 0.0

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

    def sample(self, replace: bool = False) -> Tuple[List[Transition], bool, bool]:
        batch_size = self.batch_size

        from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
        from_beta = min(batch_size - int(self.priority_fraction * batch_size), len(self.beta_memory))
        res = random.sample(self.alpha_memory, from_alpha) + random.sample(self.beta_memory, from_beta)
        random.shuffle(res)
        return res, False, False

    def update_beta(self, step: int):
        pass

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)
