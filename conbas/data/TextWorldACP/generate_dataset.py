import argparse
from dataclasses import dataclass
from glob import glob
from tqdm import tqdm
import csv
import numpy as np
import random
from typing import List, Set, Optional
from pathlib import Path


@dataclass
class Stats:
    lines: int = 0
    states: int = 0
    cmds: int = 0
    unique_cmds: int = 0
    mean_cmds_per_state: float = 0.0
    std_cmds_per_state: float = 0.0


def set_rng_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    return np.random.default_rng(seed)


def stats_to_str(s: Stats, title: str, print_lines=False, not_in_train: Optional[Set] = None) -> str:
    stats_str = f"{title}:\n"
    stats_str += "=" * len(f"{title}:") + "\n"

    if print_lines:
        stats_str += f"Lines:                {s.lines}\n"

    stats_str += f"States:               {s.states}\n"
    stats_str += f"Commands:             {s.cmds}\n"
    stats_str += f"Unique Commands:      {s.unique_cmds}\n"
    stats_str += f"Commands per State:   {s.mean_cmds_per_state:.2f} (std. {s.std_cmds_per_state:.4f})\n"

    if not_in_train is not None:
        stats_str += f"Command not in Train: {len(not_in_train)}\n"

    return stats_str


def create_dataset(rows: List[List[str]]):
    stats = Stats()

    total_commands = set()
    n_commands = []
    for row in rows:
        admissible_cmds = row[1:]
        total_commands.update(admissible_cmds)
        n_commands.append(len(admissible_cmds))

    # stats
    n_commands = np.array(n_commands)
    stats.states = len(rows)
    stats.unique_cmds = len(total_commands)
    stats.cmds = n_commands.sum()
    stats.mean_cmds_per_state = n_commands.mean()
    stats.std_cmds_per_state = n_commands.std()

    return rows, stats, total_commands


def parse_args():
    parser = argparse.ArgumentParser(description="Generates TextWorld ACP dataset.")
    parser.add_argument("log_dir", type=str)
    parser.add_argument("-out_dir", type=str, default="./data/TextWorldACP")
    parser.add_argument("-train_split", type=float, default=.85)
    parser.add_argument("-valid_split", type=float, default=.1)
    parser.add_argument("-seed", type=int, default=2_183_154_691)
    return parser.parse_args()


def main():
    args = parse_args()

    rng = set_rng_seed(args.seed)

    stats = Stats()
    files = glob(args.log_dir + "/**/*_states.txt", recursive=True)

    states_cmds_mapping = {}

    # gather all states
    for file in files:
        with open(file, "r") as fp:
            lines = fp.readlines()
        stats.lines += len(lines)

        reader = csv.reader(lines)
        with tqdm(reader) as pbar:
            for row in pbar:
                state, admissible_cmds = row[0], set(row[1:])

                if state not in states_cmds_mapping:
                    states_cmds_mapping[state] = admissible_cmds
                else:
                    states_cmds_mapping[state].update(admissible_cmds)

    # order all admissible commands
    rows = []
    total_commands = set()
    n_commands = []
    for state, admissible_cmds in states_cmds_mapping.items():
        total_commands.update(admissible_cmds)
        n_commands.append(len(admissible_cmds))
        rows.append([state, *sorted(list(admissible_cmds))])

    total_commands = sorted(list(total_commands))

    # stats
    n_commands = np.array(n_commands)
    stats.states = len(rows)
    stats.unique_cmds = len(total_commands)
    stats.cmds = n_commands.sum()
    stats.mean_cmds_per_state = n_commands.mean()
    stats.std_cmds_per_state = n_commands.std()

    # split dataset in train/valid/test
    test_split = 1 - (args.train_split + args.valid_split)
    assert(args.train_split > 0.0 and args.valid_split > 0.0 and test_split > 0.0)
    total_size = len(rows)
    train_size = int(total_size * args.train_split)
    valid_size = int(total_size * args.valid_split)
    test_size = total_size - train_size - valid_size
    assert(train_size > 0 and valid_size > 0 and test_size > 0)

    permutation = rng.permutation(np.arange(total_size))

    rows_train, stats_train, cmds_train = create_dataset([rows[idx] for idx in permutation[:train_size]])
    rows_valid, stats_valid, cmds_valid = create_dataset([rows[idx] for idx in permutation[train_size:train_size+valid_size]])
    rows_test, stats_test, cmds_test = create_dataset([rows[idx] for idx in permutation[train_size+valid_size:]])
    assert(len(rows_train) == train_size and len(rows_valid) == valid_size and len(rows_test) == test_size)

    # write datasets
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    with open(out_dir / "train.txt", "w") as fp:
        writer = csv.writer(fp)
        writer.writerows(rows_train)

    with open(out_dir / "valid.txt", "w") as fp:
        writer = csv.writer(fp)
        writer.writerows(rows_valid)

    with open(out_dir / "test.txt", "w") as fp:
        writer = csv.writer(fp)
        writer.writerows(rows_test)

    # write commands
    with open(out_dir / "commands.txt", "w") as fp:
        fp.write("\n".join(total_commands))

    stats_str = ""
    stats_str += stats_to_str(stats, "Total", True)

    stats_str += "\n" + stats_to_str(stats_train, "Train")
    stats_str += "\n" + stats_to_str(stats_valid, "Valid", not_in_train=cmds_valid.difference(cmds_train))
    stats_str += "\n" + stats_to_str(stats_test, "Test", not_in_train=cmds_test.difference(cmds_train))

    with open(out_dir / "stats.txt", "w") as fp:
        fp.write(stats_str)

    print(stats_str)


if __name__ == "__main__":
    main()
