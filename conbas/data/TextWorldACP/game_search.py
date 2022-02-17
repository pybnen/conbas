from typing import List
from pathlib import Path
from general import extract_state, State, get_environment, \
    state_eq_wo_feedback, log_state
import csv
from glob import glob
import argparse
from tqdm import tqdm
from copy import copy
import random


all_commands = set()
visited: List[State] = []


def append_visited(state: State):
    global visited, csv_writer
    log_state(csv_writer, state)
    visited.append(state)


def has_visited(state: State):
    global visited
    filtered = [s for s in visited if state_eq_wo_feedback(state, s)]
    assert len(filtered) <= 1
    return len(filtered) != 0


def backtrack(env, path):
    env.reset()
    for command in path:
        env.step(command)


def filter_cmds(cmd):
    return cmd.find("examine") != 0 and cmd != "inventory" and cmd != "look"
    # return cmd != "inventory" and cmd != "look"


def depth_first_search_aux(env, state, admissible_commands, path, max_depth, branching_factor):
    global all_commands, visited

    if has_visited(state):
        return

    all_commands.update(admissible_commands)
    append_visited(state)

    if state.done or max_depth != -1 and len(path) > max_depth:
        return

    commands = list(filter(filter_cmds, admissible_commands))
    if branching_factor != -1 and len(commands) > branching_factor:
        commands = random.sample(commands, k=branching_factor)

    for command in commands:
        path.append(command)
        ob, _, done, infos = env.step(command)
        next_game_state = (ob, infos, done)
        next_state, next_admissible_commands = extract_state(next_game_state)

        if not has_visited(next_state):
            depth_first_search_aux(env, next_state, next_admissible_commands, copy(path),
                                   max_depth, branching_factor)

        # reset prev state
        path.pop()
        backtrack(env, path)


def depth_first_search(env, max_depth=-1, branching_factor=-1):
    path = []
    ob, infos = env.reset()
    game_state = (ob, infos, False)
    state, admissible_commands = extract_state(game_state)

    depth_first_search_aux(env, state, admissible_commands, path, max_depth,
                           branching_factor)


def depth_first_search_iter(env, max_depth=-1, branching_factor=-1):
    global visited, all_commands

    path = []
    ob, infos = env.reset()
    game_state = (ob, infos, False)
    state, admissible_commands = extract_state(game_state)

    stack = []
    stack.append((state, admissible_commands, copy(path)))

    while len(stack) != 0:
        state, admissible_commands, path = stack.pop()

        if has_visited(state):
            continue

        all_commands.update(admissible_commands)
        append_visited(state)

        if state.done or max_depth != -1 and len(path) > max_depth:
            continue

        backtrack(env, path)

        commands = list(filter(filter_cmds, admissible_commands))
        if branching_factor != -1 and len(commands) > branching_factor:
            commands = random.sample(commands, k=branching_factor)

        for command in commands:
            path.append(command)
            ob, _, done, infos = env.step(command)
            next_game_state = (ob, infos, done)
            next_state, next_admissible_commands = extract_state(next_game_state)

            if not has_visited(next_state):
                stack.append((next_state, next_admissible_commands, copy(path)))

            # reset prev state
            path.pop()
            backtrack(env, path)


def breadth_first_search(env, max_depth=-1, branching_factor=-1):
    global visited, all_commands

    path = []
    ob, infos = env.reset()
    game_state = (ob, infos, False)
    state, admissible_commands = extract_state(game_state)

    queue = []
    queue.append((state, admissible_commands, copy(path)))

    while len(queue) != 0:
        state, admissible_commands, path = queue.pop(0)

        if has_visited(state):
            continue

        all_commands.update(admissible_commands)
        append_visited(state)

        if state.done or max_depth != -1 and len(path) > max_depth:
            continue

        backtrack(env, path)

        commands = list(filter(filter_cmds, admissible_commands))
        if branching_factor != -1 and len(commands) > branching_factor:
            commands = random.sample(commands, k=branching_factor)

        for command in commands:
            path.append(command)
            ob, _, done, infos = env.step(command)
            next_game_state = (ob, infos, done)
            next_state, next_admissible_commands = extract_state(next_game_state)

            if not has_visited(next_state):
                queue.append((next_state, next_admissible_commands, copy(path)))

            # reset prev state
            path.pop()
            backtrack(env, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generates TextWorld ACP dataset.")
    parser.add_argument("game_dir", type=str)
    parser.add_argument("-log_dir", type=str, required=True)
    parser.add_argument("-max_depth", type=int, default=-1)
    parser.add_argument("-branching_factor", type=int, default=-1)
    parser.add_argument("-seed", type=int, default=2_183_154_691)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)

    game_files = sorted(glob(args.game_dir + "**/*.ulx", recursive=True))
    logdir = Path(args.log_dir)
    logdir.mkdir(parents=True, exist_ok=True)

    with tqdm(game_files) as pbar:
        for game_file in pbar:
            pbar.set_description(game_file)
            logfile = logdir / (Path(game_file).name + "_states.txt")
            with open(logfile, "w") as log_fp:
                csv_writer = csv.writer(log_fp)
                env = get_environment(game_file)

                visited = []
                all_commands = set()
                # depth_first_search(env, max_depth=args.max_depth, branching_factor=args.branching_factor)
                depth_first_search_iter(env, max_depth=args.max_depth, branching_factor=args.branching_factor)
                # breadth_first_search(env,
                #                     max_depth=args.max_depth,
                #                     branching_factor=args.branching_factor)

                pbar.set_postfix({"states": len(visited)})
                with open(logdir / (Path(game_file).name + "_cmds.txt"), "w") as cmd_fp:
                    cmd_fp.write("\n".join(all_commands))
