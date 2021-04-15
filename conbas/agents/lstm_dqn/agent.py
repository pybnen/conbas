from typing import NamedTuple, Optional, List, Dict, Any, Tuple, Callable
import math
import random
from pathlib import Path
from collections import deque

import spacy
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_

from textworld import EnvInfos

from model import LstmDqnModel
from policy import LinearAnnealedEpsGreedyQPolicy
from my_utils import preprocess, words_to_ids


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


# TODO impl from microsft start code, read paper and reimplement it
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


class LstmDqnAgent:
    MODEL_CKPT_SUBFOLDER: str = "saved_models"

    def __init__(self, config: Dict[str, Any], commands: List[str], word_vocab: List[str]) -> None:
        self.commands = commands
        self.config = config
        self.prev_commands = []
        self.word_vocab = word_vocab
        # TODO maybe do this in train method / also handly on exist (i.e. should be overwritten?)
        self.experiment_path = Path(config["checkpoint"]["experiments_path"]) / config["checkpoint"]["experiment_tag"]
        self.experiment_path.mkdir(parents=True, exist_ok=True)

        self.lstm_dqn = LstmDqnModel(self.config["model"], commands, word_vocab)
        self.lstm_dqn_target = LstmDqnModel(self.config["model"], commands, word_vocab)

        # TODO this would be the place to load a pretrained model

        # copy parameter from model to target model
        self.update_target_model(tau=1.0)

        # self.lstm_dqn_target.load_state_dict(self.lstm_dqn.state_dict())
        for target_parameter, parameter in zip(self.lstm_dqn_target.parameters(), self.lstm_dqn.parameters()):
            assert torch.allclose(target_parameter.data, parameter.data)

        # self.policy = EpsGreedyQPolicy(self.config["general"]["eps"])
        self.policy = LinearAnnealedEpsGreedyQPolicy(**self.config["general"]["linear_anneald_args"])

        self.tokenizer = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])

        self.word2id: Dict[str, int] = {}
        for i, w in enumerate(self.word_vocab):
            self.word2id[w] = i

    @ staticmethod
    def request_infos() -> Optional[EnvInfos]:
        request_infos = EnvInfos()
        request_infos.description = True
        request_infos.inventory = True
        request_infos.entities = True
        request_infos.verbs = True
        request_infos.admissible_commands = True
        request_infos.command_templates = True
        return request_infos

    def init(self, obs: List[str], infos: Dict[str, List[Any]]) -> None:
        self.prev_commands = ["" for _ in range(len(obs))]

    def pad_input_ids(self, input_ids: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensor_list = [torch.tensor(item) for item in input_ids]
        input_tensor = pad_sequence(input_tensor_list, padding_value=self.word2id["<PAD>"])
        input_lengths = torch.tensor([len(seq) for seq in input_tensor_list])
        return input_tensor, input_lengths

    def extract_input(self, obs: List[str],
                      infos: Dict[str, List[Any]]) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
        """Extracts DQN network input, from current state information

        Args:
            obs: List that contains the current observation (=feedback) for each game
            infos: additional (step) information for each game

        returns:
            input_tensor: tensor of shape (max_len, batch_size) containing the padded state information
                for each game in the batch.
                max_len: maximum length of game information
                batch_size: number of games in batch
            input_lengths: tensor of shape (batch_size) containing the length of each input sequence
            input_ids: list of sequences containing the ids that describe the input
        """
        inventory_tokens = [preprocess(item, self.tokenizer) for item in infos["inventory"]]
        inventory_ids = [words_to_ids(tokens, self.word2id) for tokens in inventory_tokens]

        observation_tokens = [preprocess(item, self.tokenizer) for item in obs]
        observation_ids = [words_to_ids(tokens, self.word2id) for tokens in observation_tokens]

        prev_command_tokens = [preprocess(item, self.tokenizer) for item in self.prev_commands]
        prev_command_ids = [words_to_ids(tokens, self.word2id) for tokens in prev_command_tokens]

        look_tokens = [preprocess(item, self.tokenizer) for item in infos["description"]]
        for i, l in enumerate(look_tokens):
            if len(l) == 0:
                look_tokens[i] = ["end"]
        look_ids = [words_to_ids(tokens, self.word2id) for tokens in look_tokens]

        input_ids = [_l + i + o + pc for _l, i, o, pc in zip(look_ids,
                                                             inventory_ids,
                                                             observation_ids,
                                                             prev_command_ids)]
        input_tensor, input_lengths = self.pad_input_ids(input_ids)
        return input_tensor, input_lengths, input_ids

    def q_values(self, input_tensor, input_lengths, lstm_dqn):
        state_representations = lstm_dqn.representation_generator(input_tensor, input_lengths)
        return self.lstm_dqn.command_scorer(state_representations)

    def act(self, obs: List[str], infos: Dict[str, List[Any]]
            ) -> Tuple[List[str], torch.Tensor, List[List[int]]]:
        input_tensor, input_lengths, input_ids = self.extract_input(obs, infos)

        q_values = self.q_values(input_tensor, input_lengths, self.lstm_dqn)

        # command selection
        command_indices = self.policy.select_command(q_values)

        commands = [self.commands[i] for i in command_indices]
        self.prev_commands = commands

        return commands, command_indices.detach(), input_ids

    def save_state_dict(self) -> None:
        ckpt_dir = self.experiment_path / self.MODEL_CKPT_SUBFOLDER
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.lstm_dqn.state_dict(), ckpt_dir / "model_weights.pt")

    def load_state_dict(self, load_from) -> None:
        state_dict = torch.load(load_from)
        self.lstm_dqn.load_state_dict(state_dict)
        # TODO do i have to do this?
        # copy parameter from model to target model
        self.update_target_model(tau=1.0)

    def train(self, env, train_config) -> None:
        n_episodes = train_config["n_episodes"]
        batch_size = train_config["batch_size"]

        update_after = train_config["update_after"]
        tau = train_config["soft_update_tau"]
        discount = train_config["discount"]

        # save_fequency = self.config["checkpoint"]["save_frequency"]

        # TODO maybe use huber loss (= smooth_l1_loss)
        mse = nn.MSELoss(reduction="mean")

        # TODO use prio replay memory
        replay_memory = ReplayMemory(capacity=train_config["replay_capacity"],
                                     batch_size=train_config["replay_batch_size"])
        clip_grad_norm = train_config["optimizer"]["clip_grad_norm"]

        parameters = filter(lambda p: p.requires_grad,
                            self.lstm_dqn.parameters())
        optimizer = optim.Adam(parameters, lr=train_config["optimizer"]["lr"])

        maxlen = 150
        mov_scores = deque(maxlen=maxlen)
        mov_steps = deque(maxlen=maxlen)
        mov_losses = deque(maxlen=maxlen)

        n_batches = int(math.ceil(n_episodes / batch_size))
        with tqdm(range(n_batches))as pbar:
            for _ in pbar:
                self.lstm_dqn.train()
                obs, infos = env.reset()
                self.init(obs, infos)

                scores = np.array([0] * len(obs))
                dones = [False] * len(obs)
                not_or_recently_dones = [True] * len(obs)
                steps = [0] * len(obs)

                while not all(dones):
                    steps = [step + int(not done)
                             for step, done in zip(steps, dones)]

                    commands, command_indices, input_ids = self.act(obs, infos)

                    old_scores = scores
                    obs, scores, dones, infos = env.step(commands)

                    # calculate immediate reward from scores
                    rewards = np.array(scores) - old_scores

                    _, _, next_input_ids = self.extract_input(obs, infos)

                    for i, (input_id, command_index, reward, next_input_id, done) \
                            in enumerate(zip(input_ids, command_indices, rewards, next_input_ids, dones)):

                        # only append transitions from not done or just recently done episodes
                        if not_or_recently_dones[i]:
                            if done:
                                not_or_recently_dones[i] = False

                            replay_memory.append(
                                Transition(input_id, command_index, reward, next_input_id, done))

                    if len(replay_memory) > replay_memory.batch_size and len(replay_memory) > update_after:
                        loss = self.update(discount, replay_memory, mse)
                        optimizer.zero_grad()
                        # TODO check what retrain_graph param does
                        loss.backward(retain_graph=False)
                        clip_grad_norm_(self.lstm_dqn.parameters(), clip_grad_norm)
                        optimizer.step()

                        self.update_target_model(tau)

                        mov_losses.append(loss.detach().item())

                # TODO do i really need this?
                # Let the agent knows the game is done.
                self.act(obs, infos)

                mov_scores.extend(scores)
                mov_steps.extend(steps)

                pbar.set_postfix({
                    "eps": self.policy.eps,
                    "score": np.mean(mov_scores),
                    "steps": np.mean(mov_steps),
                    "loss": np.mean(mov_losses)})

        # TODO Add timer
        print("Done")

    def update(self,
               discount: float,
               replay_memory: Memory,
               loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        transitions = replay_memory.sample()

        # This is a neat trick to convert a batch transitions into one
        # transition that contains in each attribute a batch of that attribute,
        # found here: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        # and explained in more detail here: https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip/19343#19343
        batch = TransitionBatch(*zip(*transitions))  # type: ignore

        # create tensors for update
        input_tensor, input_lengths = self.pad_input_ids(batch.observation)
        command_indices = torch.stack(batch.command_index, dim=0)
        rewards = torch.tensor(batch.reward)
        next_input_tensor, next_input_lengths = self.pad_input_ids(batch.next_observation)
        non_terminal_mask = 1.0 - torch.tensor(batch.done, dtype=torch.float32)

        # q_values from policy network, Q(obs, a, phi)
        q_values = self.q_values(input_tensor, input_lengths, self.lstm_dqn).gather(
            dim=1, index=command_indices.unsqueeze(-1)).squeeze(-1)

        # argmax_a Q(next_obs, a, phi)
        _, argmax_a = self.q_values(next_input_tensor, next_input_lengths, self.lstm_dqn).max(dim=1)
        # Q(next_obs, argmax_a Q(next_obs, a, phi), phi_minus)
        next_q_values = self.q_values(next_input_tensor, next_input_lengths, self.lstm_dqn_target).gather(
            dim=1, index=argmax_a.unsqueeze(-1)).squeeze(-1)
        # target = reward + discount * Q(next_obs, argmax_a Q(next_obs, a, phi), phi_minus) * non_terminal_mask
        target = rewards + non_terminal_mask * discount * next_q_values.detach()

        loss = loss_fn(q_values, target)

        return loss

    def update_target_model(self, tau):
        for target_parameter, parameter in zip(self.lstm_dqn_target.parameters(), self.lstm_dqn.parameters()):
            target_parameter.data.copy_(tau * parameter.data + (1.0 - tau) * target_parameter.data)

    # def hard_update_target_model(self):
    #     self.target_net_update_freq = 10
    #     self.update_count = (self.update_count + 1) % self.target_net_update_freq
    #     if self.update_count == 0:
    #         self.lstm_dqn_target.load_state_dict(self.lstm_dqn.state_dict())
