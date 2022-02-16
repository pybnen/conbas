from copy import copy
from typing import List
from pathlib import Path
from general import extract_state, State, get_environment, \
    state_eq_wo_feedback, log_state
import csv
from glob import glob

logdir = Path(f"./dfs/logs_csv/")
logdir.mkdir(parents=True, exist_ok=True)
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
    # return cmd.find("examine") != 0 and cmd != "inventory" and cmd != "look"   
    return cmd != "inventory" and cmd != "look"   
   
   
def depth_first_search_iter(env):
    global visited
    
    all_commands = set()
   
    ob, infos = env.reset()
    game_state = (ob, infos, False)
    path = []
    state, admissible_commands = extract_state(game_state)
    
    stack = []
    stack.append((state, admissible_commands, copy(path)))
    
    while len(stack) != 0:
        state, admissible_commands, path = stack.pop()
        backtrack(env, path)
        
        all_commands.update(admissible_commands)
        
        if has_visited(state):
            continue
        
        # append to visited
        append_visited(state)
            
        if state.done:
            continue
        
        filtered_commands = list(filter(filter_cmds, admissible_commands))
        for command in filtered_commands:
            path.append(command)
            ob, _, done, infos = env.step(command)
            
            next_game_state = (ob, infos, done)
            next_state, next_admissible_commands = extract_state(next_game_state)

            if not has_visited(next_state):
                stack.append((next_state, next_admissible_commands, copy(path)))
            
            # reset prev state
            path.pop()
            backtrack(env, path)
    
    return all_commands
   
   
if __name__ == "__main__":
    game_dir = "./SaladWorld/"
    game_files = sorted(glob(game_dir + "**/*.ulx", recursive=True))
    
    for game_file in game_files:
        print(game_file)
        logfile = logdir / (Path(game_file).name + "_states.txt")
        with open(logfile, "w") as log_fp:
            csv_writer = csv.writer(log_fp)
            env = get_environment(game_file)
            visited = []
            
            commands = depth_first_search_iter(env)

            with open(logdir / (Path(game_file).name + "_cmds.txt"), "w") as cmd_fp:
                cmd_fp.write("\n".join(commands))




