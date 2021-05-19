import argparse

import numpy as np
import torch

import textworld
import textworld.gym
import gym


def parse_arguments():
    description = "Play a TextWorld game (.z8 or .ulx)."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("game_file")
    parser.add_argument("ckpt_path")
    parser.add_argument("--cpu", default=None, action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, metavar="STEPS",
                        help="Limit maximum number of steps.")
    args = parser.parse_args()
    return args


def get_agent(args):
    from conbas.agents.lstm_dqn.train import get_commands, get_word_vocab
    from conbas.agents import LstmDqnAgent
    from conbas.agents.lstm_dqn.policy import EpsGreedyQPolicy

    device = None
    if args.cpu:
        device = torch.device("cpu")

    ckpt = torch.load(args.ckpt_path, map_location=device)
    config = ckpt["config"]
    # overwrite cuda setting if argument is set
    if args.cpu:
        config["general"]["use_cuda"] = False

    commands_files = config["general"]["commands_files"]
    vocab_file = config["general"]["vocab_file"]

    commands = get_commands(commands_files)
    word_vocab = get_word_vocab(vocab_file)

    agent = LstmDqnAgent(config, commands, word_vocab)
    agent.policy = EpsGreedyQPolicy(0.1, agent.device)
    agent.load_state_dict(args.ckpt_path, "state_dict")
    agent.lstm_dqn.train()
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
    args = parse_arguments()

    # create agent
    agent = get_agent(args)

    requested_infos = agent.request_infos()
    requested_infos.max_score = True
    env_id = textworld.gym.register_game(args.game_file,
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
