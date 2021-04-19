from typing import Optional, List, Dict, Any, Tuple, Callable
import math
from pathlib import Path
from collections import deque
import time
import shutil
from datetime import datetime
import socket

import spacy
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

import gym
from textworld import EnvInfos

from model import LstmDqnModel
from policy import LinearAnnealedEpsGreedyQPolicy
from utils import preprocess, words_to_ids
from core import Memory, ReplayMemory, Transition, TransitionBatch


class LstmDqnAgent:
    CONFIG_FILENAME = "config.py"
    MODEL_CKPT_SUBFOLDER: str = "saved_models"
    ON_EXIST_IGNORE = "ignore"
    ON_EXIST_DELETE = "delete"
    ON_EXIST_ERR = "err"

    def __init__(self, config: Dict[str, Any], commands: List[str], word_vocab: List[str]) -> None:
        self.commands = commands
        self.config = config
        self.prev_commands = []
        self.word_vocab = word_vocab
        self.experiment_path = None
        self.lstm_dqn = LstmDqnModel(self.config["model"], commands, word_vocab)
        self.lstm_dqn_target = LstmDqnModel(self.config["model"], commands, word_vocab)

        # NOTE: This would be the place to load a pretrained model

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
        """Request the infos the agent expects from the environment

        Returns:
            request_infos: EnvInfos"""
        request_infos = EnvInfos()
        request_infos.description = True
        request_infos.inventory = True
        request_infos.entities = True
        request_infos.verbs = True
        request_infos.admissible_commands = True
        request_infos.command_templates = True
        return request_infos

    def init(self, obs: List[str], infos: Dict[str, List[Any]]) -> None:
        """Init some properties for a new episodes

        Args:
            obs: List that contains the current observation (=feedback) for each game
            infos: additional (step) information for each game
        """
        self.prev_commands = ["" for _ in range(len(obs))]

    def pad_input_ids(self, input_ids: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad a list of sequences, the sequence contains ids that are the input for the agent

        Args:
            input_ids: list of sequences containing the ids that describe the input

        Returns:
            input_tensor: padded tensor of input ids
            input_lengths: lenght of each unpadded sequence
        """
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

        Returns:
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

    def q_values(self, input_tensor: torch.Tensor, input_lengths: torch.Tensor, lstm_dqn: LstmDqnModel) -> torch.Tensor:
        """Calculate Q values for all commands for the current input

        Args:
            input_tensor: tensor of shape (max_len, batch_size) containing the padded state information
            input_lengths: tensor of shape (batch_size) containing the length of each input sequence
            lstm_dqn: LstmDqnModel to calculate q value

        Returns:
            command_scores: tensor fo shape (batch_size, n_commands) for each state a list
                containing the scores of each command in that state
        """
        state_representations = lstm_dqn.representation_generator(input_tensor, input_lengths)
        return self.lstm_dqn.command_scorer(state_representations)

    def act(self, obs: List[str], infos: Dict[str, List[Any]]
            ) -> Tuple[List[str], torch.Tensor, List[List[int]]]:
        """Returns command for the current observation and information from the environment.

        Args:
            obs: List that contains the current observation (=feedback) for each game
            infos: additional (step) information for each game

        Returns:
            commands: List of commands, one command per game in the batch
            command_indices: tensor of shape (batch_size, ), contains the command index for each state
            input_ids: list of sequences containing the ids that describe the input
        """
        input_tensor, input_lengths, input_ids = self.extract_input(obs, infos)

        q_values = self.q_values(input_tensor, input_lengths, self.lstm_dqn)

        # command selection
        command_indices = self.policy.select_command(q_values)

        commands = [self.commands[i] for i in command_indices]
        self.prev_commands = commands

        return commands, command_indices.detach(), input_ids

    def save_state_dict(self, filename: str) -> None:
        """Save state dict to experiment path.

        Args:
            filename: the name of the file
        """
        if self.experiment_path is not None:
            ckpt_dir = self.experiment_path / self.MODEL_CKPT_SUBFOLDER
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            torch.save(self.lstm_dqn.state_dict(), ckpt_dir / filename)
        else:
            print("Experiment path not defined, make sure to call .train(env, train_config) before.")

    def load_state_dict(self, load_from) -> None:
        """Load state dict.

        Args:
            load_from: path to state dict file
        """
        state_dict = torch.load(load_from)
        self.lstm_dqn.load_state_dict(state_dict)
        # copy parameter from model to target model
        self.update_target_model(tau=1.0)

    def _setup_experiment(self, ckpt_config: Dict[str, Any]) -> None:
        """Create directory for experiment.

        Args:
            ckpt_config: checkpoint configurations
        """
        self.experiment_path = Path(
            ckpt_config["experiments_path"]) / ckpt_config["experiment_tag"]

        on_exist = ckpt_config["on_exist"]
        if on_exist == self.ON_EXIST_IGNORE:
            self.experiment_path.mkdir(parents=True, exist_ok=True)
        elif on_exist == self.ON_EXIST_ERR:
            self.experiment_path.mkdir(parents=True, exist_ok=False)
        elif on_exist == self.ON_EXIST_DELETE:
            if self.experiment_path.exists():
                shutil.rmtree(self.experiment_path)
            self.experiment_path.mkdir(parents=True, exist_ok=False)

        # TODO here all relevant files/information for training could be stored
        # copy config file
        shutil.copyfile(Path(__file__).parent / self.CONFIG_FILENAME, self.experiment_path / self.CONFIG_FILENAME)

        log_dir = self.experiment_path / \
            "{}_{}".format(datetime.now().strftime("%b%d_%H-%M-%S"),
                           socket.gethostname())
        # log_dir.mkdir(exist_ok=False)
        self.writer = SummaryWriter(log_dir)

    def train(self, env: gym.Env) -> None:
        """Train the model on the given environment.

        Args:
            env: game envrionment, can contain multiple games, that will be played in batches
        """
        train_config: Dict[str, Any] = self.config["training"]

        try:
            self._setup_experiment(self.config["checkpoint"])
        except FileExistsError:
            print("Experiment dir already exists, maybe change tag or set on_exist to ignore.")
            return

        n_episodes = train_config["n_episodes"]
        batch_size = train_config["batch_size"]

        update_after = train_config["update_after"]
        tau = train_config["soft_update_tau"]
        discount = train_config["discount"]

        if train_config["loss_fn"] == "smooth_l1":
            loss_fn = nn.SmoothL1Loss(reduction="mean")
        else:
            # default to mse
            loss_fn = nn.MSELoss(reduction="mean")

        replay_memory = ReplayMemory(capacity=train_config["replay_capacity"],
                                     batch_size=train_config["replay_batch_size"])
        clip_grad_norm = train_config["optimizer"]["clip_grad_norm"]

        parameters = filter(lambda p: p.requires_grad,
                            self.lstm_dqn.parameters())
        optimizer = optim.Adam(parameters, lr=train_config["optimizer"]["lr"])

        maxlen = 500
        mov_scores = deque(maxlen=maxlen)
        mov_steps = deque(maxlen=maxlen)
        mov_losses = deque(maxlen=maxlen)

        start_time = time.time()
        save_frequency = self.config["checkpoint"]["save_frequency"]
        n_batches = int(math.ceil(n_episodes / batch_size))
        global_step = 0
        try:
            with tqdm(range(1, n_batches + 1)) as pbar:
                for batch_num in pbar:
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
                            loss = self.update(discount, replay_memory, loss_fn)
                            optimizer.zero_grad()
                            loss.backward(retain_graph=False)
                            clip_grad_norm_(self.lstm_dqn.parameters(), clip_grad_norm)
                            optimizer.step()

                            self.update_target_model(tau)

                            # save train statistics
                            mov_losses.append(loss.detach().item())
                            self.writer.add_scalar("train/loss", loss.detach().item(), global_step=global_step)
                            global_step += 1

                    # display/save statistics
                    mov_scores.extend(scores)
                    mov_steps.extend(steps)

                    self.writer.add_scalar("train/score", np.mean(scores), global_step=batch_num)
                    self.writer.add_scalar("train/steps", np.mean(steps), global_step=batch_num)
                    self.writer.add_scalar("general/epsilon", self.policy.eps, global_step=batch_num)
                    
                    pbar.set_postfix({
                        "eps": self.policy.eps,
                        "score": np.mean(mov_scores),
                        "steps": np.mean(mov_steps),
                        "loss": np.mean(mov_losses)})

                    # save model
                    if batch_num % save_frequency == 0:
                        self.save_state_dict("model_weights_{}.pt".format(batch_num * batch_size))

        except KeyboardInterrupt:
            print("Keyboard Interrupt")

        print("Done, execution time: {}sec.".format(time.time() - start_time))

    def update(self,
               discount: float,
               replay_memory: Memory,
               loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Prepares one update step.

        Samples transition batch from replay memory, and calculates loss.

        Args:
            discount: gamma
            replay_memory: contains transition history
            loss_fn: used to calculate loss
        Returns:
            loss: loss of the transition batch
        """
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

    def update_target_model(self, tau: float):
        """Performs soft update of target network

        Args:
            tau: dictates how much the target network is updated with the policy network
        """
        for target_parameter, parameter in zip(self.lstm_dqn_target.parameters(), self.lstm_dqn.parameters()):
            target_parameter.data.copy_(tau * parameter.data + (1.0 - tau) * target_parameter.data)

    # def hard_update_target_model(self):
    #     self.target_net_update_freq = 10
    #     self.update_count = (self.update_count + 1) % self.target_net_update_freq
    #     if self.update_count == 0:
    #         self.lstm_dqn_target.load_state_dict(self.lstm_dqn.state_dict())
