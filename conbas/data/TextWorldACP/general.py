import re
from dataclasses import dataclass
import textworld
from textworld.envs.wrappers import Filter
from typing import List, Optional

INTRO = "________ ________ __ __ ________ | \\| \\| \\ | \\| \\ \\$$$$$$$$| $$$$$$$$| $$ | $$ \\$$$$$$$$ | $$ | $$__ \\$$\\/ $$ | $$ | $$ | $$ \\ >$$ $$ | $$ | $$ | $$$$$ / $$$$\\ | $$ | $$ | $$_____ | $$ \\$$\\ | $$ | $$ | $$ \\| $$ | $$ | $$ \\$$ \\$$$$$$$$ \\$$ \\$$ \\$$ __ __ ______ _______ __ _______ | \\ _ | \\ / \\ | \\ | \\ | \\ | $$ / \\ | $$| $$$$$$\\| $$$$$$$\\| $$ | $$$$$$$\\ | $$/ $\\| $$| $$ | $$| $$__| $$| $$ | $$ | $$ | $$ $$$\\ $$| $$ | $$| $$ $$| $$ | $$ | $$ | $$ $$\\$$\\$$| $$ | $$| $$$$$$$\\| $$ | $$ | $$ | $$$$ \\$$$$| $$__/ $$| $$ | $$| $$_____ | $$__/ $$ | $$$ \\$$$ \\$$ $$| $$ | $$| $$ \\| $$ $$ \\$$ \\$$ \\$$$$$$ \\$$ \\$$ \\$$$$$$$$ \\$$$$$$$"


@dataclass
class State:
    description: str
    inventory: str
    feedback: str
    last_command: str
    admissible_commands: List[str]
    done: bool
    commands: Optional[List[str]] = None


@dataclass
class Transition:
    s1: State
    s2: State


def preprocess(text: str, type: str = None) -> str:
    # merge all spaces (whitespace, tab, linefeed)
    # to a single whitespace
    text = re.sub("\s+", " ", text)

    if type == 'feedback':
        # remove ascii art from text
        text = text.replace(INTRO, "")

    # strip beginning/ending spaces
    text = text.strip()
    return text


def extract_state(game_state):
    infos = game_state[1]

    description = preprocess(infos['description'])
    inventory = preprocess(infos['inventory'])
    feedback = preprocess(infos['feedback'], type='feedback')
    last_command = preprocess(infos['last_command'] or "")
    admissible_commands = [preprocess(cmd)
                           for cmd in infos['admissible_commands']]
    done = game_state[2]

    return State(description,
                 inventory,
                 feedback,
                 last_command,
                 admissible_commands,
                 done), admissible_commands


def state_eq(s1: State, s2: State):
    return s1.description == s2.description and \
        s1.inventory == s2.inventory and \
        s1.feedback == s2.feedback


def state_eq_wo_feedback(s1: State, s2: State):
    return s1.description == s2.description and \
        s1.inventory == s2.inventory and \
        s1.admissible_commands == s2.admissible_commands


def transition_eq(t1: Transition, t2: Transition):
    return state_eq_wo_feedback(t1.s1, t2.s1) and \
        state_eq_wo_feedback(t1.s2, t2.s2) and \
        t1.s2.admissible_commands == t2.s2.admissible_commands


def log_transition(csv_writer, t: Transition):
    state = f"{t.s1.description} {t.s1.inventory} "
    state += f"{t.s2.last_command} "
    state += f"{t.s2.feedback} {t.s2.description} {t.s2.inventory}"
    row = [state, *t.s2.admissible_commands]
    csv_writer.writerow(row)


def log_state(csv_writer, s: State):
    state_str = f"{s.description} {s.inventory}"
    row = [state_str, *s.admissible_commands]
    csv_writer.writerow(row)


def get_environment(file, additional_infos=False):
    env_infos = textworld.EnvInfos()
    env_infos.admissible_commands = True
    env_infos.feedback = True
    env_infos.inventory = True
    env_infos.description = True
    env_infos.last_command = True
    if additional_infos:
        env_infos.extras = ["walkthrough"]

    wrappers = [Filter]
    env = textworld.start(file, infos=env_infos, wrappers=wrappers)
    return env


def state_to_json(s: State):
    state_str = f"{s.description} {s.inventory}"
    return [state_str, s.admissible_commands, s.commands]
