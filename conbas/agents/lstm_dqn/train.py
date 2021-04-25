from typing import List

import gym
import textworld
import textworld.gym

from config import config
from agent import LstmDqnAgent


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


def train():
    game_files: List[str] = config["general"]["game_files"]
    commands_files: List[str] = config["general"]["commands_files"]
    vocab_file: str = config["general"]["vocab_file"]

    commands = get_commands(commands_files)
    word_vocab = get_word_vocab(vocab_file)

    agent = LstmDqnAgent(config, commands, word_vocab)
    requested_infos = agent.request_infos()
    env_id = textworld.gym.register_games(game_files,
                                          requested_infos,
                                          batch_size=config["training"]["batch_size"],
                                          asynchronous=True, auto_reset=False,
                                          max_episode_steps=config["training"]["max_steps_per_episode"],
                                          name="training")
    env = gym.make(env_id)
    agent.train(env)

    # obs, infos = env.reset()
    # obs = ["The End Is Never... ", "The End."]
    # infos = {
    #    "inventory": ["", ""],
    #    "description": ["", "In the room is nothing."],
    #    "extra.recipe": ["", ""]
    # }
    # batch_size = len(obs)
    # scores = [0] * batch_size
    # dones = [False] * batch_size
    # agent.init(obs, infos)

    # step 1
    # commands = agent.act(obs, scores, dones, infos)
    # for o, c in zip(obs, commands):
    #     print(o)
    #     print(c)
    # obs, scores, dones, infos = env.step(commands)

    # step 2
    # commands = agent.act(obs, scores, dones, infos)
    # for o, c in zip(obs, commands):
    #     print(o)
    #     print(c)
    # obs, scores, dones, infos = env.step(commands)


if __name__ == "__main__":
    train()
