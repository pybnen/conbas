from typing import List, NamedTuple, Tuple, Callable

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

        self.stats = {"reward_mean": 0.0, "reward_total": 0.0, "reward_cnt": {}}

    def append(self, transition: Transition) -> None:
        # calculate priorization
        self.priorities[self.pos] = np.max(self.priorities) if len(self) > 0 else 1.0

        # add transition to memory
        old_reward = 0.0
        if len(self) < self.capacity:
            reward_total = self.stats["reward_total"] + transition.reward
            self.memory.append(transition)
        else:
            old_reward = self.memory[self.pos].reward
            reward_total = self.stats["reward_total"] + transition.reward - old_reward
            self.memory[self.pos] = transition

        # add new reward count
        if transition.reward != 0.0:
            if transition.reward not in self.stats["reward_cnt"]:
                self.stats["reward_cnt"][transition.reward] = 1
            else:
                self.stats["reward_cnt"][transition.reward] += 1

        # remove old reward count
        if old_reward != 0.0:
            self.stats["reward_cnt"][old_reward] -= 1

        self.stats["reward_total"] = reward_total
        self.stats["reward_mean"] = reward_total / len(self)

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
