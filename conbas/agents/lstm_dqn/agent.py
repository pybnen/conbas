from typing import Optional, List, Dict, Any, Tuple, Callable
from pathlib import Path
from collections import deque
import time
import shutil
from datetime import datetime
import socket
# from torch.types import Number
import yaml

import spacy
from tqdm import tqdm

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

import gym
from textworld import EnvInfos

from .model import LstmDqnModel
from .policy import AnnealedEpsGreedyQPolicy
from .utils import preprocess, words_to_ids, linear_decay_fn, linear_inc_fn
from .core import Transition, TransitionBatch, PrioritizedReplayMemory


class LstmDqnAgent:
    MODEL_CKPT_SUBFOLDER: str = "saved_models"
    ON_EXIST_IGNORE = "ignore"
    ON_EXIST_DELETE = "delete"
    ON_EXIST_ERR = "err"

    def __init__(self, config: Dict[str, Any], commands: List[str], word_vocab: List[str]) -> None:
        assert not config["training"]["use_double_DQN"] or config["training"]["use_target_network"]
        
        self.commands = commands
        self.config = config
        use_cuda = torch.cuda.is_available() and config["general"]["use_cuda"]
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.prev_commands = []
        self.word_vocab = word_vocab
        self.experiment_path = None
        self.lstm_dqn = LstmDqnModel(self.config["model"], commands, word_vocab, self.device).to(self.device)

        # NOTE: This would be the place to load a pretrained model

        if self.config["training"]["use_target_network"]:
            self.lstm_dqn_target = LstmDqnModel(self.config["model"], commands, word_vocab, self.device).to(self.device)
            # set target network to eval mode
            self.lstm_dqn_target.eval()

            # copy parameter from model to target model
            self.update_target_model(tau=1.0)
            # self.lstm_dqn_target.load_state_dict(self.lstm_dqn.state_dict())
            
            for target_parameter, parameter in zip(self.lstm_dqn_target.parameters(), self.lstm_dqn.parameters()):
                assert torch.allclose(target_parameter.data, parameter.data)
        else:
            self.lstm_dqn_target = self.lstm_dqn

        annealed_args = self.config["general"]["eps_annealed_args"]
        self.anneal_fn = linear_decay_fn(**annealed_args)
        self.policy = AnnealedEpsGreedyQPolicy(
            annealed_args["upper_bound"], device=self.device, anneal_fn=self.anneal_fn)

        # self.policy = EpsGreedyQPolicy(self.config["general"]["eps"])
        # policy_config = {"device": self.device, **self.config["general"]["linear_annealed_args"]}
        # self.policy = LinearAnnealedEpsGreedyQPolicy(**policy_config)

        self.tokenizer = spacy.load('en_core_web_sm')

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
        request_infos.max_score = True
        request_infos.objective = True
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
        input_tensor = pad_sequence(input_tensor_list, padding_value=self.word2id["<PAD>"]).to(self.device)
        input_lengths = torch.tensor([len(seq) for seq in input_tensor_list])
        return input_tensor, input_lengths

    def extract_input(self, obs: List[str],
                      infos: Dict[str, List[Any]],
                      prev_commands: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
        """Extracts DQN network input, from current state information

        Args:
            obs: List that contains the current observation (=feedback) for each game
            infos: additional (step) information for each game
            prev_commands: previous command for each game

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

        prev_command_tokens = [preprocess(item, self.tokenizer) for item in prev_commands]
        prev_command_ids = [words_to_ids(tokens, self.word2id) for tokens in prev_command_tokens]

        quest_tokens = [preprocess(item, self.tokenizer) for item in infos["objective"]]
        quest_id_list = [words_to_ids(tokens, self.word2id) for tokens in quest_tokens]

        look_tokens = [preprocess(item, self.tokenizer) for item in infos["description"]]
        for i, l in enumerate(look_tokens):
            if len(l) == 0:
                look_tokens[i] = ["end"]
        look_ids = [words_to_ids(tokens, self.word2id) for tokens in look_tokens]

        input_ids = [_l + i + q + o + pc for _l, i, q, o, pc in zip(look_ids,
                                                             inventory_ids,
                                                             quest_id_list,
                                                             observation_ids,
                                                             prev_command_ids)]
        input_tensor, input_lengths = self.pad_input_ids(input_ids)
        return input_tensor, input_lengths, input_ids

    def q_values(self, input_tensor: torch.Tensor, input_lengths: torch.Tensor,
                 lstm_dqn_model: LstmDqnModel) -> torch.Tensor:
        """Calculate Q values for all commands for the current input

        Args:
            input_tensor: tensor of shape (max_len, batch_size) containing the padded state information
            input_lengths: tensor of shape (batch_size) containing the length of each input sequence
            lstm_dqn: LstmDqnModel to calculate q value

        Returns:
            command_scores: tensor fo shape (batch_size, n_commands) for each state a list
                containing the scores of each command in that state
        """
        state_representations = lstm_dqn_model.representation_generator(input_tensor, input_lengths)
        return lstm_dqn_model.command_scorer(state_representations)

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
        input_tensor, input_lengths, input_ids = self.extract_input(obs, infos, self.prev_commands)

        # no need to build a computation graph here
        with torch.no_grad():
            q_values = self.q_values(input_tensor, input_lengths, self.lstm_dqn)
        assert not q_values.requires_grad

        # command selection
        command_indices = self.policy.select_command(q_values)
        assert not command_indices.requires_grad

        commands = [self.commands[i] for i in command_indices]
        self.prev_commands = commands

        return commands, command_indices, input_ids

    def save_checkpoint(self, filename: str) -> None:
        """Save checkpoint to experiment path.

        Args:
            filename: the name of the file
        """
        if self.experiment_path is None:
            print("Experiment path not defined, make sure to call .train(env, train_config) before.")

        ckpt_dir = self.experiment_path / self.MODEL_CKPT_SUBFOLDER
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "state_dict": self.lstm_dqn.state_dict(),
            "config": self.config
        }

        torch.save(ckpt, ckpt_dir / filename)

    def load_state_dict(self, load_from, key=None) -> None:
        """Load state dict.

        Args:
            load_from: path to state dict file
        """
        state_dict = torch.load(load_from, map_location=self.device)
        if key is not None:
            state_dict = state_dict[key]

        self.lstm_dqn.load_state_dict(state_dict)
        # copy parameter from model to target model
        self.update_target_model(tau=1.0)

    def _setup_experiment(self, ckpt_config: Dict[str, Any], config_file: str, beta_anneal_fn) -> None:
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
        config_file_path = Path(config_file)
        shutil.copyfile(config_file, self.experiment_path / config_file_path.name)
        with open(self.experiment_path / "agent_config.yaml", "w") as fp:
            fp.write(yaml.dump(self.config))

        expected_update_steps = self.config["training"]["traning_steps"] // self.config["training"]["batch_size"]

        # plot eps decay
        fig, ax = plt.subplots(figsize=(15, 7))
        plt.xlabel("expected update steps")
        plt.ylabel("eps")
        ax.plot([self.anneal_fn(s) for s in range(0, expected_update_steps)])
        fig.savefig(self.experiment_path / "annealed_eps.png")
        plt.close(fig)

        # plot beta anneal
        fig, ax = plt.subplots(figsize=(15, 7))
        plt.xlabel("expected update steps")
        plt.ylabel("beta")
        ax.plot([beta_anneal_fn(s) for s in range(0, expected_update_steps)])
        fig.savefig(self.experiment_path / "annealed_beta.png")
        plt.close(fig)

        log_dir = self.experiment_path / \
            "{}_{}".format(datetime.now().strftime("%b%d_%H-%M-%S"),
                           socket.gethostname())
        # log_dir.mkdir(exist_ok=False)
        self.writer = SummaryWriter(log_dir)

    def train(self, env: gym.Env, config_file: str) -> None:
        """Train the model on the given environment.

        Args:
            env: game envrionment, can contain multiple games, that will be played in batches
        """
        train_config: Dict[str, Any] = self.config["training"]

        # setup replay buffer -------------------------------------------------
        buffer_args = train_config["replay_buffer"]
        anneald_args = train_config["replay_buffer"]["beta_annealed_args"]
        anneal_fn = linear_inc_fn(**anneald_args)
        replay_memory = PrioritizedReplayMemory(capacity=buffer_args["capacity"],
                                                batch_size=buffer_args["batch_size"],
                                                alpha=buffer_args["alpha"],
                                                start_beta=anneald_args["lower_bound"],
                                                anneal_fn=anneal_fn)

        # setup experiment directory ------------------------------------------
        try:
            self._setup_experiment(self.config["checkpoint"], config_file, anneal_fn)
        except FileExistsError:
            print("Experiment dir already exists, maybe change tag or set on_exist to ignore.")
            return

        # define training params  ---------------------------------------------
        max_training_steps = train_config["traning_steps"]

        update_after = train_config["update_after"]
        update_per_k_game_steps = train_config["update_per_k_game_steps"]

        tau = train_config["target_update_tau"]
        target_update_interval = train_config["target_update_interval"]

        discount = train_config["discount"]
        step_limit = train_config["max_steps_per_episode"]

        if train_config["loss_fn"] == "smooth_l1":
            loss_fn = nn.SmoothL1Loss(reduction="none")
        else:
            # default to mse
            loss_fn = nn.MSELoss(reduction="none")

        clip_grad_norm = train_config["optimizer"]["clip_grad_norm"]

        parameters = filter(lambda p: p.requires_grad,
                            self.lstm_dqn.parameters())
        optimizer = optim.Adam(parameters, lr=train_config["optimizer"]["lr"])

        maxlen = 100
        score_avg = deque(maxlen=maxlen)
        step_avg = deque(maxlen=maxlen)
        loss_avg = deque(maxlen=maxlen)
        grad_norm_avg = deque(maxlen=maxlen)

        save_frequency = self.config["checkpoint"]["save_frequency"]

        # start training  -----------------------------------------------------
        start_time = time.time()
        update_step = 0
        old_traning_steps = 0
        training_steps = 0   # env interactions
        epoch = 0

        try:
            with tqdm(range(1, max_training_steps + 1)) as pbar:
                while training_steps < max_training_steps:
                    self.lstm_dqn.train()
                    obs, infos = env.reset()
                    self.init(obs, infos)

                    scores = np.array([0] * len(obs))
                    max_scores = np.array(infos["max_score"])
                    steps = [0] * len(obs)
                    current_game_step = 0
                    losses = []
                    grad_norms = []

                    # distinguish between:
                    # - done:                       env won/lost
                    # - step_limit_reached:         env not won/lost
                    # - finished:                   done or step_limit_reached
                    # - nor_or_recently_finished    either not finished or finished this step
                    dones = [False] * len(obs)
                    step_limit_reached = [False] * len(obs)
                    batch_finished = [False] * len(obs)
                    not_or_recently_finished = [True] * len(obs)

                    while not all(batch_finished):
                        # increment step only if env is not finished
                        steps = [step + int(not finished) for step, finished in zip(steps, batch_finished)]

                        # count interactions with environment
                        training_steps += sum([int(not finished) for finished in batch_finished])

                        commands, command_indices, input_ids = self.act(obs, infos)
                        # move command_indices to cpu befor storing them in replay buffer
                        command_indices = command_indices.cpu()

                        old_scores = scores
                        obs, scores, dones, infos = env.step(commands)

                        # calculate immediate reward from scores and normalize it
                        rewards = (np.array(scores) - old_scores) / max_scores
                        rewards = np.array(rewards, dtype=np.float32)

                        _, _, next_input_ids = self.extract_input(obs, infos, self.prev_commands)

                        step_limit_reached = [step >= step_limit for step in steps]
                        batch_finished = [done or reached for done, reached in zip(dones, step_limit_reached)]

                        for i, (input_id, command_index, reward, next_input_id, done, finished) \
                                in enumerate(
                                    zip(input_ids, command_indices, rewards, next_input_ids, dones, batch_finished)):

                            # only append transitions from not finished or just recently finished episodes
                            if not_or_recently_finished[i]:
                                if finished:
                                    not_or_recently_finished[i] = False
                                # done is True only if env is won/lost, not if step limit is reached
                                replay_memory.append(
                                    Transition(input_id, command_index, reward, next_input_id, done))

                        if len(replay_memory) > replay_memory.batch_size and len(replay_memory) >= update_after \
                                and current_game_step % update_per_k_game_steps == 0:
                            loss, total_norm = self.update(
                                discount, replay_memory, loss_fn, optimizer, clip_grad_norm)

                            # save train statistics
                            losses.append(loss)
                            grad_norms.append(total_norm)

                            update_step += 1
                            # update alpha/beta/epsilon
                            self.update_hyperparameter(update_step, replay_memory)

                            # update target model
                            if train_config["use_target_network"] and update_step % target_update_interval == 0:
                                self.update_target_model(tau)

                            # save model
                            if update_step % save_frequency == 0:
                                self.save_checkpoint("model_weights_{}.pt".format(update_step))

                        current_game_step += 1

                        if training_steps >= max_training_steps:
                            break  # while not finished

                    score_avg.append(np.mean(scores))  # scores contains final scores
                    step_avg.append(np.mean(steps))
                    loss_avg.append(np.mean(losses))
                    grad_norm_avg.append(np.mean(grad_norms))

                    # display/save statistics
                    self.writer.add_scalar('avg_score', np.mean(score_avg) / max_scores[0], training_steps)
                    self.writer.add_scalar('curr_score', score_avg[-1] / max_scores[0], training_steps)

                    self.writer.add_scalar('avg_step', np.mean(step_avg), training_steps)
                    self.writer.add_scalar('curr_step', step_avg[-1], training_steps)

                    self.writer.add_scalar('avg_loss', np.mean(loss_avg), training_steps)
                    self.writer.add_scalar('curr_loss', loss_avg[-1], training_steps)

                    self.writer.add_scalar("curr_gradient_total_norm", np.mean(grad_norm_avg),
                                           global_step=training_steps)
                    self.writer.add_scalar("curr_gradient_total_norm", grad_norm_avg[-1], global_step=training_steps)

                    self.writer.add_scalar("general/beta", replay_memory.beta, global_step=training_steps)
                    self.writer.add_scalar("general/epsilon", self.policy.eps, global_step=training_steps)

                    pbar.update(n=training_steps - old_traning_steps)
                    old_traning_steps = training_steps

                    pbar.set_postfix({
                        "epoch": epoch,
                        "eps": self.policy.eps,
                        "score": np.mean(score_avg) / max_scores[0],
                        "steps": np.mean(step_avg),
                        "loss": np.mean(loss_avg)})

                    epoch += 1

        except KeyboardInterrupt:
            print("Keyboard Interrupt")

        print("Done, execution time: {}sec.".format(time.time() - start_time))

    def update(self,
               discount: float,
               replay_memory: PrioritizedReplayMemory,
               loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
               optimizer: optim.Optimizer,
               clip_grad_norm: float):  # -> Tuple[Number, Number]:
        assert self.lstm_dqn.training

        transitions, weights, indices = replay_memory.sample()

        # This is a neat trick to convert a batch transitions into one
        # transition that contains in each attribute a batch of that attribute,
        # found here: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        # and explained in more detail here: https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip/19343#19343
        batch = TransitionBatch(*zip(*transitions))  # type: ignore

        # create tensors for update
        input_tensor, input_lengths = self.pad_input_ids(batch.observation)
        command_indices = torch.stack(batch.command_index, dim=0).to(self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_input_tensor, next_input_lengths = self.pad_input_ids(batch.next_observation)
        non_terminal_mask = 1.0 - torch.tensor(batch.done, dtype=torch.float32, device=self.device)
        weights = torch.from_numpy(weights).to(device=self.device)

        # q_values from policy network, Q(obs, a, phi)
        q_values = self.q_values(input_tensor, input_lengths, self.lstm_dqn).gather(
            dim=1, index=command_indices.unsqueeze(-1)).squeeze(-1)
        assert q_values.requires_grad

        # no need to build a computation graph here
        with torch.no_grad():
            if self.config["training"]["use_double_DQN"]:
                # argmax_a Q(next_obs, a, phi)
                _, argmax_a = self.q_values(next_input_tensor, next_input_lengths, self.lstm_dqn).max(dim=1)
                # Q(next_obs, argmax_a Q(next_obs, a, phi), phi_minus)
                next_q_values = self.q_values(next_input_tensor, next_input_lengths, self.lstm_dqn_target).gather(
                    dim=1, index=argmax_a.unsqueeze(-1)).squeeze(-1)
                assert not next_q_values.requires_grad

            else:
                # in this case lstm_dqn === lstm_dqn
                next_q_values, _ = self.q_values(next_input_tensor, next_input_lengths, self.lstm_dqn_target).max(dim=1)

        # target = reward + discount * Q(next_obs, argmax_a Q(next_obs, a, phi), phi_minus) * non_terminal_mask
        target = rewards + non_terminal_mask * discount * next_q_values.detach()

        loss = loss_fn(q_values, target) * weights
        priorities = loss.detach().cpu().numpy()
        loss = loss.mean()

        # update step
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        total_norm = clip_grad_norm_(self.lstm_dqn.parameters(), clip_grad_norm)
        optimizer.step()

        # update priorities
        replay_memory.update_priorities(indices, priorities)

        # in server torch version total_norm is float
        if type(total_norm) == torch.Tensor:
            total_norm = total_norm.detach().cpu().item()

        return loss.detach().cpu().item(), total_norm

    def update_target_model(self, tau: float):
        """Performs soft update of target network

        Args:
            tau: dictates how much the target network is updated with the policy network
        """
        for target_parameter, parameter in zip(self.lstm_dqn_target.parameters(), self.lstm_dqn.parameters()):
            target_parameter.data.copy_(tau * parameter.data + (1.0 - tau) * target_parameter.data)

    def update_hyperparameter(self, update_step, replay_memory):
        self.policy.update(update_step)
        replay_memory.update_beta(update_step)
