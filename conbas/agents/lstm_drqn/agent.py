from typing import Optional, List, Dict, Any, Tuple, Callable
from pathlib import Path
from collections import deque
import time
import shutil
from datetime import datetime
import socket
from torch.types import Number
# from torch.types import Number
import yaml
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

from .model import LstmDrqnModel
from ..helpers.policy import AnnealedEpsGreedyQPolicy
from ..helpers.utils import preproc, words_to_ids, linear_decay_fn, linear_inc_fn
from .core import Transition, TransitionBatch, PrioritizedReplayMemory, CCPrioritizedReplayMemory, PrioritizedSequenceReplayMemory


class LstmDrqnAgent:
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
        self.lstm_dqn = LstmDrqnModel(self.config["model"], commands, word_vocab, self.device).to(self.device)

        # NOTE: This would be the place to load a pretrained model

        if self.config["training"]["use_target_network"]:
            self.lstm_dqn_target = LstmDrqnModel(self.config["model"], commands, word_vocab, self.device).to(self.device)
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

        self.word2id: Dict[str, int] = {}
        for i, w in enumerate(self.word_vocab):
            self.word2id[w] = i

        self._setup_counting_reward()

    def _setup_counting_reward(self) -> None:
        self.state_cnt = None
        beta = self.config["general"]["counting_reward"]["beta"]

        def episodic_discovery_bonus(state_str: List[str]) -> List[Number]:
            batch_size = len(state_str)
            count_rewards = []

            for i in range(batch_size):
                if state_str[i] not in self.state_cnt[i]:
                    self.state_cnt[i][state_str[i]] = 1
                else:
                    self.state_cnt[i][state_str[i]] += 1

                cnt = self.state_cnt[i][state_str[i]]
                reward = beta * float(cnt == 1)
                count_rewards.append(reward)

            return count_rewards

        def cumulative_counting_bonus(state_str: List[str]) -> List[Number]:
            batch_size = len(state_str)
            count_rewards = []

            for i in range(batch_size):
                if state_str[i] not in self.state_cnt:
                    self.state_cnt[state_str[i]] = 1
                else:
                    self.state_cnt[state_str[i]] += 1

                cnt = self.state_cnt[state_str[i]]
                reward = beta * cnt**(-1/3)
                count_rewards.append(reward)

            return count_rewards

        def _reset_episodic(batch_size):
            self.state_cnt = [{} for _ in range(batch_size)]

        def _reset_cumulative():
            self.state_cnt = {}

        if self.config["general"]["counting_reward"]["type"] == "episodic":
            self.reset_state_cnt = _reset_episodic
            self.reset_state_cnt(self.config["training"]["batch_size"])

            self.get_counting_reward = episodic_discovery_bonus
        elif self.config["general"]["counting_reward"]["type"] == "cumulative":
            self.reset_state_cnt = _reset_cumulative
            self.reset_state_cnt()

            self.get_counting_reward = cumulative_counting_bonus
        else:
            self.get_counting_reward = lambda state: [0.0] * len(state)

    # def reset_state_cnt(self, batch_size):
    #     self.state_cnt = [{} for _ in range(batch_size)]

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
            input_tensor: tensor of shape (max_len, batch_size) containing the padded state information
                for each game in the batch.
                max_len: maximum length of game information
                batch_size: number of games in batch
            input_lengths: tensor of shape (batch_size) containing the length of each input sequence
        """
        input_tensor_list = [torch.tensor(item) for item in input_ids]
        input_tensor = pad_sequence(input_tensor_list, padding_value=self.word2id["<PAD>"]).to(self.device)
        input_lengths = torch.tensor([len(seq) for seq in input_tensor_list])
        return input_tensor, input_lengths

    def extract_state_str(self, infos: Dict[str, List[Any]]) -> List[str]:
        inventory_strings = infos["inventory"]
        description_strings = infos["description"]
        observation_strings = [_d + _i for (_d, _i) in zip(description_strings, inventory_strings)]
        return observation_strings

    def extract_input(self, obs: List[str],
                      infos: Dict[str, List[Any]],
                      prev_commands: List[str]) -> List[List[int]]:
        """Extracts DQN network input, from current state information

        Args:
            obs: List that contains the current observation (=feedback) for each game
            infos: additional (step) information for each game
            prev_commands: previous command for each game

        Returns:
            input_ids: list of sequences containing the ids that describe the input
        """
        inventory_tokens = [preproc(item, str_type='inventory', lower_case=True) for item in infos["inventory"]]
        inventory_ids = [words_to_ids(tokens, self.word2id) for tokens in inventory_tokens]

        observation_tokens = [preproc(item, str_type='feedback', lower_case=True) for item in obs]
        observation_ids = [words_to_ids(tokens, self.word2id) for tokens in observation_tokens]

        prev_command_tokens = [preproc(item, str_type='None', lower_case=True) for item in prev_commands]
        prev_command_ids = [words_to_ids(tokens, self.word2id) for tokens in prev_command_tokens]

        quest_tokens = [preproc(item, str_type='None', lower_case=True) for item in infos["objective"]]
        quest_id_list = [words_to_ids(tokens, self.word2id) for tokens in quest_tokens]

        look_tokens = [preproc(item, str_type='description', lower_case=True) for item in infos["description"]]
        for i, l in enumerate(look_tokens):
            if len(l) == 0:
                look_tokens[i] = ["end"]
        look_ids = [words_to_ids(tokens, self.word2id) for tokens in look_tokens]

        input_ids = [_l + i + q + o + pc for _l, i, q, o, pc in zip(look_ids,
                                                                    inventory_ids,
                                                                    quest_id_list,
                                                                    observation_ids,
                                                                    prev_command_ids)]
        # input_tensor, input_lengths = self.pad_input_ids(input_ids)
        return input_ids

    def q_values(self, input_tensor: torch.Tensor, input_lengths: torch.Tensor,
                 lstm_dqn_model: LstmDrqnModel,
                 h_0: torch.Tensor = None,
                 c_0: torch.Tensor = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Calculate Q values for all commands for the current input

        Args:
            input_tensor: tensor of shape (max_len, batch_size) containing the padded state information
            input_lengths: tensor of shape (batch_size) containing the length of each input sequence
            lstm_dqn: LstmDqnModel to calculate q value
            h_0:
            c_0:

        Returns:
            command_scores: tensor fo shape (batch_size, n_commands) for each state a list
                containing the scores of each command in that state
        """
        state_representations = lstm_dqn_model.representation_generator(input_tensor, input_lengths)
        return lstm_dqn_model.command_scorer(state_representations, h_0, c_0)

    def act(self, input_ids: List[List[int]],
            h_0: torch.Tensor = None,
            c_0: torch.Tensor = None) -> Tuple[List[str], torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Returns command for the current observation and information from the environment.

        Args:
            input_ids: list of sequences containing the ids that describe the input
            h_0:
            c_0:

        Returns:
            commands: List of commands, one command per game in the batch
            command_indices: tensor of shape (batch_size, ), contains the command index for each state
        """
        input_tensor, input_lengths = self.pad_input_ids(input_ids)

        # no need to build a computation graph here
        with torch.no_grad():
            q_values, (h_1, c_1) = self.q_values(input_tensor, input_lengths, self.lstm_dqn, h_0, c_0)
        assert not q_values.requires_grad

        # command selection
        command_indices = self.policy.select_command(q_values)
        assert not command_indices.requires_grad

        commands = [self.commands[i] for i in command_indices]
        self.prev_commands = commands

        return commands, command_indices, (h_1, c_1)

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
        if beta_anneal_fn is not None:
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
        anneal_fn = None

        solved_is_prior = train_config["replay_buffer"]["solved_is_prior"]
        if buffer_args["type"] == "my":
            anneald_args = train_config["replay_buffer"]["beta_annealed_args"]
            anneal_fn = linear_inc_fn(**anneald_args)
            replay_memory = PrioritizedReplayMemory(capacity=buffer_args["capacity"],
                                                    batch_size=buffer_args["batch_size"],
                                                    history_size=buffer_args["history_size"],
                                                    alpha=buffer_args["alpha"],
                                                    start_beta=anneald_args["lower_bound"],
                                                    anneal_fn=anneal_fn)
        elif buffer_args["type"] == "seq":
            anneald_args = train_config["replay_buffer"]["beta_annealed_args"]
            anneal_fn = linear_inc_fn(**anneald_args)
            replay_memory = PrioritizedSequenceReplayMemory(capacity=buffer_args["capacity"],
                                                            batch_size=buffer_args["batch_size"],
                                                            history_size=buffer_args["history_size"],
                                                            alpha=buffer_args["alpha"],
                                                            start_beta=anneald_args["lower_bound"],
                                                            anneal_fn=anneal_fn)
        elif buffer_args["type"] == "cc":
            replay_memory = CCPrioritizedReplayMemory(capacity=buffer_args["capacity"],
                                                      batch_size=buffer_args["batch_size"],
                                                      history_size=buffer_args["history_size"],
                                                      priority_fraction=buffer_args["alpha"])
        else:
            raise ValueError

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

        use_counting_reward = self.config["general"]["counting_reward"]["type"] is not False
        counting_reward_episodic = use_counting_reward \
            and self.config["general"]["counting_reward"]["type"] == 'episodic'
        lambda_annealed_args = self.config["general"]["counting_reward"]["lambda_annealed_args"]
        counting_lambda = lambda_annealed_args["upper_bound"]
        counting_lambda_fn = linear_decay_fn(**lambda_annealed_args)

        # define logging data structures --------------------------------------
        maxlen = 100
        counting_avg = deque(maxlen=maxlen)
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

                    counting_rewards = np.zeros(len(obs))
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
                    is_prior = [False] * len(obs)

                    # extract input information
                    input_ids = self.extract_input(obs, infos, self.prev_commands)
                    h, c = None, None
                    memory_cache = [[] for _ in range(len(obs))]

                    if use_counting_reward:
                        if counting_reward_episodic:
                            self.reset_state_cnt(len(obs))

                        state_str = self.extract_state_str(infos)
                        _ = self.get_counting_reward(state_str)

                    while not all(batch_finished):
                        # increment step only if env is not finished
                        steps = [step + int(not finished) for step, finished in zip(steps, batch_finished)]

                        # count interactions with environment
                        training_steps += sum([int(not finished) for finished in batch_finished])

                        commands, command_indices, (h, c) = self.act(input_ids, h, c)
                        # move command_indices to cpu befor storing them in replay buffer
                        command_indices = command_indices.cpu()

                        old_scores = scores
                        obs, scores, dones, infos = env.step(commands)

                        # calculate immediate reward from scores and normalize it
                        rewards = (np.array(scores) - old_scores) / max_scores
                        rewards = np.array(rewards, dtype=np.float32)

                        next_input_ids = self.extract_input(obs, infos, self.prev_commands)

                        step_limit_reached = [step >= step_limit for step in steps]
                        batch_finished = [done or reached for done, reached in zip(dones, step_limit_reached)]

                        if use_counting_reward:
                            new_state_str = self.extract_state_str(infos)
                            counting_reward = np.array(self.get_counting_reward(new_state_str))
                            counting_rewards += counting_reward

                            # calcualte reward including counting reward
                            rewards = rewards + counting_lambda * counting_reward

                        for i, (input_id, command_index, reward, next_input_id, done, finished) \
                                in enumerate(
                                    zip(input_ids, command_indices, rewards, next_input_ids, dones, batch_finished)):

                            # only append transitions from not finished or just recently finished episodes
                            if not_or_recently_finished[i]:
                                if finished:
                                    not_or_recently_finished[i] = False

                                # TODO make config to define is_prior
                                # 1.) one transaction as positiv reward
                                # 2.) goal reached -> done is true
                                if (solved_is_prior and done) or (not solved_is_prior and reward > 0.0):
                                    is_prior[i] = True

                                # done is True only if env is won/lost, not if step limit is reached
                                memory_cache[i].append(Transition(input_id, command_index, reward, next_input_id, done, finished))

                        input_ids = next_input_ids

                        if len(replay_memory) > replay_memory.batch_size and len(replay_memory) >= update_after \
                                and current_game_step % update_per_k_game_steps == 0:
                            loss, total_norm = self.update(
                                discount, replay_memory, loss_fn, optimizer, clip_grad_norm)

                            # save train statistics
                            losses.append(loss)
                            grad_norms.append(total_norm)

                            # update alpha/beta/epsilon
                            counting_lambda = self.update_hyperparameter(training_steps, replay_memory,
                                                                         counting_lambda_fn)

                            update_step += 1
                            # update target model
                            if train_config["use_target_network"] and update_step % target_update_interval == 0:
                                self.update_target_model(tau)

                            # save model
                            if update_step % save_frequency == 0:
                                self.save_checkpoint("model_weights_{}.pt".format(update_step))

                        current_game_step += 1

                        if training_steps >= max_training_steps:
                            break  # while not finished

                    for i, (prior, mc) in enumerate(zip(is_prior, memory_cache)):
                        if isinstance(replay_memory, PrioritizedSequenceReplayMemory):
                            replay_memory.append_episode(mc)
                        else:
                            for transition in mc:
                                replay_memory.append(transition, prior)

                    counting_avg.append(np.mean(counting_rewards))
                    score_avg.append(np.mean(scores))  # scores contains final scores
                    step_avg.append(np.mean(steps))
                    if losses:
                        loss_avg.append(np.mean(losses))
                    if grad_norms:
                        grad_norm_avg.append(np.mean(grad_norms))

                    # display/save statistics
                    self.writer.add_scalar('avg_counting_reward', np.mean(counting_avg), training_steps)
                    self.writer.add_scalar('curr_counting_reward', counting_avg[-1] / max_scores[0], training_steps)

                    self.writer.add_scalar('avg_score', np.mean(score_avg) / max_scores[0], training_steps)
                    self.writer.add_scalar('curr_score', score_avg[-1] / max_scores[0], training_steps)

                    self.writer.add_scalar('avg_step', np.mean(step_avg), training_steps)
                    self.writer.add_scalar('curr_step', step_avg[-1], training_steps)

                    if loss_avg:
                        self.writer.add_scalar('avg_loss', np.mean(loss_avg), training_steps)
                        self.writer.add_scalar('curr_loss', loss_avg[-1], training_steps)

                    if grad_norm_avg:
                        self.writer.add_scalar("curr_gradient_total_norm", np.mean(grad_norm_avg),
                                            global_step=training_steps)
                        self.writer.add_scalar("curr_gradient_total_norm", grad_norm_avg[-1], global_step=training_steps)

                    self.writer.add_scalar("general/beta", replay_memory.beta, global_step=training_steps)
                    self.writer.add_scalar("general/epsilon", self.policy.eps, global_step=training_steps)
                    self.writer.add_scalar("general/update_step", update_step, global_step=training_steps)
                    self.writer.add_scalar("general/counting_lambda", counting_lambda, global_step=training_steps)

                    # replay buffer
                    self.writer.add_scalar("replay_buffer/mean_reward",
                                           replay_memory.stats["reward_mean"], global_step=training_steps)

                    self.writer.add_scalar("replay_buffer/timeout",
                                           replay_memory.stats["timeout"], global_step=training_steps)
                    self.writer.add_scalar("replay_buffer/tries_mean",
                                           replay_memory.stats["tries_mean"], global_step=training_steps)
                    self.writer.add_scalar("replay_buffer/n_sampled",
                                           replay_memory.stats["n_sampled"], global_step=training_steps)

                    self.writer.add_scalar("replay_buffer/sampled_reward",
                                           replay_memory.stats["sampled_reward"], global_step=training_steps)
                    self.writer.add_scalar("replay_buffer/sampled_done_cnt",
                                           replay_memory.stats["sampled_done_cnt"], global_step=training_steps)
                    for r, c in replay_memory.stats["sampled_reward_cnt"].items():
                        self.writer.add_scalar("replay_buffer/sampled_cnt_{:.2f}".format(r),
                                               c, global_step=training_steps)
                    replay_memory.reset_stats()

                    pbar.update(n=training_steps - old_traning_steps)
                    old_traning_steps = training_steps

                    pbar.set_postfix({
                        "epoch": epoch,
                        "us": update_step,
                        "eps": self.policy.eps,
                        "cnt lambda": counting_lambda,
                        "cnt rew": np.mean(np.mean(counting_avg)),
                        "score": np.mean(score_avg) / max_scores[0],
                        "steps": np.mean(step_avg),
                        "loss": np.mean(loss_avg) if loss_avg else 0.0})

                    epoch += 1

        except KeyboardInterrupt:
            print("Keyboard Interrupt")

        print("Done, execution time: {}sec.".format(time.time() - start_time))

    def update(self,
               discount: float,
               replay_memory,
               loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
               optimizer: optim.Optimizer,
               clip_grad_norm: float):  # -> Tuple[Number, Number]:
        assert self.lstm_dqn.training

        def calculate_target(next_q_values, next_q_values_target, batch):
            rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
            non_terminal_mask = 1.0 - torch.tensor(batch.done, dtype=torch.float32, device=self.device)

            # no need to build a computation graph here
            with torch.no_grad():
                if self.config["training"]["use_double_DQN"]:
                    # argmax_a Q(next_obs, a, phi)
                    _, argmax_a = next_q_values.max(dim=1)
                    # Q(next_obs, argmax_a Q(next_obs, a, phi), phi_minus)
                    next_q_value = next_q_values_target.gather(dim=1, index=argmax_a.unsqueeze(-1)).squeeze(-1)
                    assert not next_q_value.requires_grad

                else:
                    # in this case lstm_dqn === lstm_dqn_target
                    next_q_value, _ = next_q_values_target.max(dim=1)
            target = rewards + non_terminal_mask * discount * next_q_value.detach()
            return target

        bootstrap_state = self.config["training"]["replay_buffer"]["bootstrap_state"]
        update_from = self.config["training"]["replay_buffer"]["update_from"]

        sequences, weights, indices = replay_memory.sample()
        # This is a neat trick to convert a batch transitions into one
        # transition that contains in each attribute a batch of that attribute,
        # found here: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        # and explained in more detail here: https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip/19343#19343
        sequence_batch = [TransitionBatch(*zip(*transitions)) for transitions in sequences]  # type: ignore

        losses = []
        priorities = []

        h, c = None, None
        h_target, c_target = None, None        

        input_tensor, input_lengths = self.pad_input_ids(sequence_batch[0].observation)
        q_values, (h, c) = self.q_values(input_tensor, input_lengths, self.lstm_dqn, h, c)
        with torch.no_grad():
            next_q_values_target, (h_target, c_target) = self.q_values(input_tensor, input_lengths,
                                                                       self.lstm_dqn_target,
                                                                       h_target, c_target)
        # TODO check off by one error
        # if bootstrap_state != 0:
        #    q_values, h, c = q_values.detach(), h.detach(), c.detach()

        for i, batch in enumerate(sequence_batch):
            # calculate q values for next state
            next_input_tensor, next_input_lengths = self.pad_input_ids(batch.next_observation)
            next_q_values, (h, c) = self.q_values(next_input_tensor, next_input_lengths, self.lstm_dqn, h, c)
            with torch.no_grad():
                next_q_values_target, (h_target, c_target) = self.q_values(next_input_tensor,
                                                                           next_input_lengths,
                                                                           self.lstm_dqn_target,
                                                                           h_target, c_target)

            # TODO check off by one error
            if i < bootstrap_state:
                q_values, h, c = q_values.detach(), h.detach(), c.detach()

            if i >= update_from:
                command_indices = torch.stack(batch.command_index, dim=0).to(self.device)
                q_value = q_values.gather(dim=1, index=command_indices.unsqueeze(-1)).squeeze(-1)
                # calculate target
                target = calculate_target(next_q_values, next_q_values_target, batch)

                loss = loss_fn(q_value, target)
                losses.append(loss)

                if self.config["training"]["replay_buffer"]["priority"] == "loss":
                    priorities.append(loss.detach().cpu().numpy())
                elif self.config["training"]["replay_buffer"]["priority"] == "td_error":
                    priorities.append((target - q_value).detach().cpu().numpy())
                else:
                    raise ValueError()

            # next q values are current y values
            q_values = next_q_values

        loss = torch.stack(losses).mean(dim=0)
        priorities = np.stack(priorities).mean(axis=0)

        if weights is not False:
            t_weights = torch.from_numpy(weights).to(device=self.device)
            loss *= t_weights

            if self.config["training"]["replay_buffer"]["priority"] == "loss":
                priorities *= weights

        loss = loss.mean()

        # update step
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        total_norm = clip_grad_norm_(self.lstm_dqn.parameters(), clip_grad_norm)
        optimizer.step()

        # update priorities
        if indices is not False:
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

    def update_hyperparameter(self, training_step, replay_memory, counting_lambda_fn):
        self.policy.update(training_step)
        replay_memory.update_beta(training_step)
        return counting_lambda_fn(training_step)
