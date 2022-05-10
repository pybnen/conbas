from typing import List

import argparse
import yaml

import gym
import textworld
import textworld.gym

# import for seeding randomness
import random
import torch
import numpy as np

from .agent import AdmacDrqnAgent

torch.backends.cudnn.benchmark = False
# try:
#     torch.use_deterministic_algorithms(True)
# except AttributeError as e:
#     print(e)


def build_parser():
    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument("config_file", help="Path to config file.")
    parser.add_argument("--tag", "-t", type=str, help="Overwrite experiment tag.")
    parser.add_argument("--desc", "-d", type=str, help="Overwrite experiment description.")
    parser.add_argument("--seed", "-s", type=int, required=True, help="Seed for pseudo-random-generation.")
    return parser


def get_commands(commands_files: List[str]) -> List[str]:
    commands = []
    for command_file in commands_files:
        with open(command_file, "r") as fp:
            for line in fp:
                line = line.strip()
                if len(line) > 0:
                    commands.append(line)
    return list(dict.fromkeys(commands))


def get_word_vocab(vocab_file: str) -> List[str]:
    with open(vocab_file) as fp:
        word_vocab = fp.read().split("\n")
    return word_vocab


def set_rng_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return np.random.default_rng(seed)


def train():
    args = build_parser().parse_args()
    np_rng = set_rng_seed(args.seed)

    with open(args.config_file, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    if args.tag is not None:
        config["checkpoint"]["experiment_tag"] = args.tag
    if args.desc is not None:
        config["checkpoint"]["experiment_description"] = args.desc

    config["general"]["seed"] = args.seed

    # print config
    print(f"Use configuration from '{args.config_file}':")
    print(yaml.dump(config))

    game_files: List[str] = config["general"]["game_files"]
    commands_files: List[str] = config["general"]["commands_files"]
    vocab_file: str = config["general"]["vocab_file"]

    commands = get_commands(commands_files)
    word_vocab = get_word_vocab(vocab_file)

    agent = AdmacDrqnAgent(config, commands, word_vocab)
    requested_infos = agent.request_infos()
    # add max score, to normalize score for logging
    requested_infos.max_score = True
    env_id = textworld.gym.register_games(game_files,
                                          requested_infos,
                                          batch_size=config["training"]["batch_size"],
                                          asynchronous=True, auto_reset=False,
                                          max_episode_steps=0,
                                          name="training")
    env = gym.make(env_id)
    agent.train(env, args.config_file)


if __name__ == "__main__":
    train()
