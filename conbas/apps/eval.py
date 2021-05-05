import argparse
import yaml
from tqdm import tqdm

import numpy as np

import torch

import textworld
import textworld.gym
import gym

from conbas.agents import RandomAgent, LstmDqnAgent


def build_parser():
    description = "Evaluate agent on a textworld environment"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("game_file")
    parser.add_argument("--agent", choices=['random', 'lstm_dqn'], required=True)
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument("--max-steps", type=int, default=0, metavar="STEPS",
                        help="Limit maximum number of steps.")
    return parser


def get_random_agent():
    def act(agent, obs, infos):
        return agent.act(obs, infos)

    print("Evaluate RandomAgent")
    return RandomAgent(use_admissible=True), act


def get_lstm_dqn_agent(ckpt_path):
    from conbas.agents.lstm_dqn.train import get_commands, get_word_vocab
    from conbas.agents.lstm_dqn.policy import EpsGreedyQPolicy

    def act(agent, obs, infos):
        with torch.no_grad():
            commands, _, _ = agent.act(obs, infos)
        return commands

    ckpt = torch.load(ckpt_path)
    config = ckpt["config"]

    print("Evaluate LstmDqnAgent")
    print(yaml.dump(config))

    commands_files = config["general"]["commands_files"]
    vocab_file = config["general"]["vocab_file"]

    commands = get_commands(commands_files)
    word_vocab = get_word_vocab(vocab_file)

    agent = LstmDqnAgent(config, commands, word_vocab)
    agent.policy = EpsGreedyQPolicy(0.1, agent.device)
    agent.load_state_dict(ckpt_path, "state_dict")
    agent.lstm_dqn.eval()
    return agent, act


def main():
    args = build_parser().parse_args()

    if args.agent == "random":
        agent, act = get_random_agent()
    elif args.agent == "lstm_dqn":
        assert args.ckpt_path is not None
        agent, act = get_lstm_dqn_agent(args.ckpt_path)
    else:
        # TODO raise exception
        assert False

    # create environment
    requested_infos = agent.request_infos()
    requested_infos.max_score = True
    env_id = textworld.gym.register_game(args.game_file,
                                         requested_infos,
                                         batch_size=1,
                                         max_episode_steps=args.max_steps)
    env = gym.make(env_id)

    obs, infos = env.reset()
    max_score = infos["max_score"][0]
    scores_arr = []

    with tqdm(range(100)) as pbar:
        for _ in pbar:
            obs, infos = env.reset()
            agent.init(obs, infos)
            scores = [0]
            dones = [False]
            n_moves = 0
            while not all(dones):  # type: ignore
                commands = act(agent, obs, infos)
                obs, scores, dones, infos = env.step(commands)
                n_moves += 1

            scores_arr.extend(scores)

            pbar.set_postfix({
                "steps": n_moves,
                "norm. score": scores[0] / max_score,
                "score": scores[0],
                "max score": max_score})

    summary = "DONE mean {:.2f}, std: {:.4f}".format(np.mean(scores_arr), np.std(scores_arr))
    print("=" * len(summary))
    print(summary)
    env.close()


if __name__ == "__main__":
    main()
