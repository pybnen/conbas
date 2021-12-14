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
    is_final: bool


class TransitionBatch(NamedTuple):
    observation: List[List[int]]
    command_index: List[torch.Tensor]
    reward: List[float]
    next_observation: List[List[int]]
    done: List[bool]
    is_final: List[bool]


class Memory:
    def __init__(self, batch_size: int, history_size: int) -> None:
        self.batch_size = batch_size
        self.history_size = history_size

    def sample(self, replace: bool = False) -> List[Transition]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class ReplayMemory(Memory):
    def __init__(self, capacity: int, batch_size: int, history_size: int) -> None:
        super().__init__(batch_size, history_size)

        self.capacity = capacity
        self.memory: List[Transition] = []
        self.pos = 0

    def append(self, transition: Transition) -> None:
        if len(self) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, replace: bool = False) -> List[List[Transition]]:
        # TODO maybe rework the sampling process

        # TODO add assertion as in PER
        
        # taken from https://github.com/xingdi-eric-yuan/TextWorld-Coin-Collector/blob/master/lstm_drqn_baseline/agent.py#L22
        batch = []
        tried_times = 0
        while len(batch) < self.batch_size:
            tried_times += 1
            if tried_times >= 500:
                break
            idx = np.random.randint(self.history_size - 1, len(self.memory) - 1)
            # only last frame can be (is_final == True)
            if np.any([item.is_final for item in self.memory[idx - (self.history_size - 1): idx]]):
                continue
            batch.append(self.memory[idx - (self.history_size - 1): idx + 1])

        if len(batch) == 0:
            return []

        batch = list(map(list, zip(*batch)))  # list (history size) of list (batch) of Transitions
        return batch

    def __len__(self) -> int:
        return len(self.memory)


class PrioritizedReplayMemory(Memory):
    def __init__(self, capacity: int, batch_size: int, history_size: int,
                 anneal_fn: Callable[[int], float], alpha: float = 0.6, start_beta: float = 0.4) -> None:
        super().__init__(batch_size, history_size)
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

    def sample(self, replace: bool = False) -> Tuple[List[List[Transition]], List[float], List[int]]:
        # TODO maybe rework the sampling process

        # taken from https://github.com/xingdi-eric-yuan/TextWorld-Coin-Collector/blob/master/lstm_drqn_baseline/agent.py#L22
        probs = self._get_distribution()

        batch = []
        indices = []
        weights = []

        tried_times = 0
        while len(batch) < self.batch_size:
            tried_times += 1
            if tried_times >= 500:
                break
            idx = np.random.choice(np.arange(len(self)), p=probs)

            if idx < self.history_size - 1 or idx >= len(self.memory) - 1:
                continue

            # only last frame can be (is_final == True)
            if np.any([item.is_final for item in self.memory[idx - (self.history_size - 1): idx]]):
                continue

            sequence = self.memory[idx - (self.history_size - 1): idx + 1]
            for i in range(len(sequence) - 1):
                assert(sequence[i].next_observation == sequence[i+1].observation)

            batch.append(sequence)
            # indices.append(list(range(idx - (self.history_size - 1), idx + 1)))
            indices.append(idx + 1)

        if len(batch) == 0:
            return [], [], []

        # weights = (len(self) * probs[np.array(indices).flatten()])**(-self.beta)
        weights = (len(self) * probs[np.array(indices)])**(-self.beta)
        weights /= np.max(weights)
        weights = np.array(weights, dtype=np.float32)
        # weights = weights.reshape(self.batch_size, self.history_size).T

        # indices = np.array(indices).T

        batch = list(map(list, zip(*batch)))  # list (history size) of list (batch) of Transitions
        return batch, weights, indices

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

    def __init__(self, capacity: int, batch_size: int, history_size: int, priority_fraction=0.0):
        super().__init__(batch_size, history_size)

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
