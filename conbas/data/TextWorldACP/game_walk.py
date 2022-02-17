import argparse
import random
from pathlib import Path
from glob import glob
from tqdm import tqdm
from general import get_environment, extract_state, State, log_state
import csv
from typing import List, Dict, Set

visited: List[State] = []
state_to_idx: Dict[str, int] = {}
all_commands: Set[str] = set()


def backtrack(env, path):
    env.reset()
    for command in path:
        env.step(command)


def filter_cmds(cmd):
    return cmd.find("examine") != 0 and cmd != "inventory" and cmd != "look"
    # return cmd != "inventory" and cmd != "look"


def get_branching_cmds(admissible, command, branching_factor):
    branching_cmds = [cmd for cmd in filter(filter_cmds, admissible)
                      if cmd != command]

    if len(branching_cmds) > branching_factor:
        branching_cmds = random.sample(branching_cmds, k=branching_factor)

    return branching_cmds


def walk_game(env, commands, branching_factor):
    global csv_writer, all_commands

    state_cnt = 1

    ob, infos = env.reset()
    state, admissible_commands = extract_state((ob, infos, False))
    log_state(csv_writer, state)

    path = []
    for command in commands:
        all_commands.update(admissible_commands)

        if branching_factor > 0:
            branching_cmds = get_branching_cmds(admissible_commands, command, branching_factor)
            for branch_cmd in branching_cmds:
                path.append(branch_cmd)

                ob, _, done, infos = env.step(branch_cmd)
                state, ac = extract_state((ob, infos, done))

                all_commands.update(ac)
                log_state(csv_writer, state)
                state_cnt += 1

                path.pop()
                backtrack(env, path)

        path.append(command)
        ob, _, done, infos = env.step(command)
        state, admissible_commands = extract_state((ob, infos, done))

        log_state(csv_writer, state)
        state_cnt += 1
    return state_cnt


def parse_args():
    parser = argparse.ArgumentParser(description="Generates TextWorld ACP dataset.")
    parser.add_argument("game_dir", type=str)
    parser.add_argument("-log_dir", type=str, required=True)
    parser.add_argument("-out_filename", type=str, default="states.txt")
    parser.add_argument("-file_ending", type=str, default="ulx")
    parser.add_argument("-branching_factor", type=int, default=3)
    parser.add_argument("-seed", type=int, default=2_183_154_691)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)

    game_files = sorted(glob(args.game_dir + f"**/*.{args.file_ending}", recursive=True))
    logdir = Path(args.log_dir)
    logdir.mkdir(parents=True, exist_ok=True)

    with open(logdir / args.out_filename, "w") as log_fp:
        csv_writer = csv.writer(log_fp)

        with tqdm(game_files) as pbar:
            for game_file in pbar:
                pbar.set_description(game_file)

                env = get_environment(game_file, True)
                ob, infos = env.reset()
                if "extra.walkthrough" not in infos or infos['extra.walkthrough'] is None:
                    continue
                commands = infos['extra.walkthrough']
                state_cnt = walk_game(env, commands, args.branching_factor)

                pbar.set_postfix({"states": state_cnt})
                log_fp.flush()

    with open(logdir / "cmds.txt", "w") as cmd_fp:
        cmd_fp.write("\n".join(all_commands))
