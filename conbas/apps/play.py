import argparse

import numpy as np
import torch

import textworld
import textworld.agents

import textworld.gym
import gym


def build_parser():
    description = "Play a TextWorld game (.z8 or .ulx)."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("game")
    parser.add_argument("ckpt_path")
    parser.add_argument("--max-steps", type=int, default=0, metavar="STEPS",
                        help="Limit maximum number of steps.")
    return parser


def get_agent(ckpt_path):
    import sys
    sys.path.insert(0, "./conbas/agents/lstm_dqn")
    from train import get_commands, get_word_vocab
    from agent import LstmDqnAgent
    from policy import EpsGreedyQPolicy

    ckpt = torch.load(ckpt_path)
    config = ckpt["config"]

    commands_files = config["general"]["commands_files"]
    vocab_file = config["general"]["vocab_file"]

    commands = get_commands(commands_files)
    word_vocab = get_word_vocab(vocab_file)

    agent = LstmDqnAgent(config, commands, word_vocab)
    agent.policy = EpsGreedyQPolicy(0.2, agent.device)
    agent.load_state_dict(ckpt_path, "state_dict")
    return agent


def render_state(obs, infos, agent):
    input_tensor, input_lengths, _ = agent.extract_input(obs, infos, agent.prev_commands)
    with torch.no_grad():
        q_values = agent.q_values(input_tensor, input_lengths, agent.lstm_dqn)

    _, argmax = q_values.max(dim=1)
    argmax = argmax.item()
    q_values = q_values.detach().cpu().numpy()
    q_values_soft = np.exp(q_values) / np.exp(q_values).sum()

    print(obs[0])
    for i, (cmd, q_value, q_value_soft) in enumerate(zip(agent.commands, q_values[0], q_values_soft[0])):
        if cmd in infos["admissible_commands"][0]:
            cmd = f"*{cmd}*"
        if argmax == i:
            cmd = f"{cmd} <-----"

        print(f"{i+1}. [{q_value:.2f}/{q_value_soft:.2f}] {cmd}")

    print("--------------------------")
    command = input(f"({agent.commands[argmax]}) > ")
    if len(command) == 0:
        command = agent.commands[argmax]

    if command.isnumeric():
        command = agent.commands[int(command) - 1]
    return command


def main():
    args = build_parser().parse_args()

    # create agent
    agent = get_agent(args.ckpt_path)

    requested_infos = agent.request_infos()
    requested_infos.max_score = True
    env_id = textworld.gym.register_game(args.game,
                                         requested_infos,
                                         batch_size=1,
                                         max_episode_steps=args.max_steps)
    env = gym.make(env_id)

    obs, infos = env.reset()
    max_score = infos["max_score"][0]
    agent.init(obs, infos)

    scores = [0]
    dones = [False]

    n_moves = 0
    while not all(dones):
        command = render_state(obs, infos, agent)
        commands = [command]
        agent.prev_commands = commands

        obs, scores, dones, infos = env.step(commands)
        n_moves += 1
    print(obs[0])

    env.close()
    print("Done after {} steps. Score {}/{}.".format(n_moves, scores[0], max_score))


if __name__ == "__main__":
    main()
