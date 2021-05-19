import argparse
import yaml
from tqdm import tqdm

import numpy as np

import torch

import textworld
import textworld.gym
import gym

from conbas.agents import RandomAgent, LstmDqnAgent


def parse_arguments():
    description = "Evaluate agent on a textworld environment"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("game_file")
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--agent", choices=['random', 'lstm_dqn'], required=True)
    parser.add_argument("--max-steps", type=int, default=50, metavar="STEPS",
                        help="Limit maximum number of steps. (0: unlimited)")

    agent_parser = {}
    agent_parser["random"] = argparse.ArgumentParser(description="Arguments for random agent.")
    agent_parser["random"].add_argument("--commands-file", type=str)

    agent_parser["lstm_dqn"] = argparse.ArgumentParser(description="Arguments for lstm_dqn agent.")
    agent_parser["lstm_dqn"].add_argument("--ckpt-path", type=str, required=True)

    args, extras = parser.parse_known_args()
    # add agent arguments to parser arguments
    agent_parser[args.agent].parse_args(extras, namespace=args)

    return args


def get_random_agent(args):
    def act(agent, obs, infos):
        return agent.act(obs, infos)

    print("Evaluate RandomAgent with arguments {}".format(args))
    if args.commands_file is not None:
        from conbas.agents.lstm_dqn.train import get_commands
        commands_files = [args.commands_file]
        commands = get_commands(commands_files)

        agent = RandomAgent(commands=commands)
    else:
        agent = RandomAgent(use_admissible=True)

    return agent, act


def get_lstm_dqn_agent(args):
    from conbas.agents.lstm_dqn.train import get_commands, get_word_vocab
    from conbas.agents.lstm_dqn.policy import EpsGreedyQPolicy

    def act(agent, obs, infos):
        with torch.no_grad():
            commands, _, _ = agent.act(obs, infos)
        return commands

    print("Evaluate LstmDqnAgent with arguments {}".format(args))

    ckpt = torch.load(args.ckpt_path)
    config = ckpt["config"]
    print(yaml.dump(config))

    commands_files = config["general"]["commands_files"]
    vocab_file = config["general"]["vocab_file"]

    commands = get_commands(commands_files)
    word_vocab = get_word_vocab(vocab_file)

    agent = LstmDqnAgent(config, commands, word_vocab)
    agent.policy = EpsGreedyQPolicy(0.1, agent.device)
    agent.load_state_dict(args.ckpt_path, "state_dict")
    agent.lstm_dqn.eval()
    return agent, act


def main():
    args = parse_arguments()

    if args.agent == "random":
        agent, act = get_random_agent(args)
    elif args.agent == "lstm_dqn":
        agent, act = get_lstm_dqn_agent(args)
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

    with tqdm(range(args.repeat)) as pbar:
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

    mean_score = np.mean(scores_arr)
    norm_score = mean_score / max_score

    summary = "DONE mean (abs/normalised) {:.2f} / {:.2f}, std: {:.4f}".format(mean_score, norm_score, np.std(scores_arr))
    print("=" * len(summary))
    print(summary)
    env.close()


if __name__ == "__main__":
    main()
