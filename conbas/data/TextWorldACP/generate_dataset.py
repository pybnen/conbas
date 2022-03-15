import argparse
from dataclasses import dataclass
from glob import glob
from tqdm import tqdm
import numpy as np
import random
from typing import Any, List, Set, Optional, Dict, Tuple
from pathlib import Path
import json


@dataclass
class Stats:
    lines: int = 0
    states: int = 0
    cmds: int = 0
    unique_cmds: int = 0
    mean_cmds_per_state: float = 0.0
    std_cmds_per_state: float = 0.0
    max_cmd_list: int = 0


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
    stats_str += f"Max Commandlist:      {s.max_cmd_list:.2f}\n"

    if not_in_train is not None:
        stats_str += f"Command not in Train: {len(not_in_train)}\n"

    return stats_str


def parse_files(files: List[str]) -> Any:
    states_map: Dict[str, Tuple[str, Set[str], Set[str]]] = {}
    stats = Stats()
    from_file: Dict[str, List[str]] = {}

    # gather all states
    for file in files:
        with open(file, "r") as fp:
            lines = fp.readlines()
        stats.lines += len(lines)

        file_stem = Path(file).stem
        from_file[file_stem] = []

        with tqdm(lines) as pbar:
            for line in pbar:
                ob, admissible_cmds, cmds = json.loads(line)
                admissible_cmds = set(admissible_cmds)
                cmds = set(cmds)

                if ob not in states_map:
                    states_map[ob] = (ob, admissible_cmds, cmds)
                    from_file[file_stem].append(ob)
                else:
                    states_map[ob][1].update(admissible_cmds)
                    states_map[ob][2].update(cmds)

    # order all admissible commands
    rows = []
    total_commands = set()
    n_commands = []
    for _, (ob, admissible_cmds, cmds) in states_map.items():
        total_commands.update(admissible_cmds)
        n_commands.append(len(admissible_cmds))
        stats.max_cmd_list = max(stats.max_cmd_list, len(cmds))
        rows.append([ob,
                     sorted(list(admissible_cmds)),
                     sorted(list(cmds))])

    total_commands = sorted(list(total_commands))

    # stats
    n_commands = np.array(n_commands)
    stats.states = len(rows)
    stats.unique_cmds = len(total_commands)
    stats.cmds = n_commands.sum()
    stats.mean_cmds_per_state = n_commands.mean()
    stats.std_cmds_per_state = n_commands.std()

    return rows, total_commands, stats, from_file


def split_rows(rows, split, rng):
    train_split, valid_split = split
    test_split = 1 - (train_split + valid_split)

    assert(train_split > 0.0 and valid_split > 0.0 and test_split > 0.0)
    total_size = len(rows)
    train_size = int(total_size * train_split)
    valid_size = int(total_size * valid_split)
    test_size = total_size - train_size - valid_size
    assert(train_size > 0 and valid_size > 0 and test_size > 0)

    permutation = rng.permutation(np.arange(total_size))

    rows_train, stats_train, cmds_train = create_dataset([rows[idx] for idx in permutation[:train_size]])
    rows_valid, stats_valid, cmds_valid = create_dataset([rows[idx]
                                                          for idx in permutation[train_size:train_size+valid_size]])
    rows_test, stats_test, cmds_test = create_dataset([rows[idx] for idx in permutation[train_size+valid_size:]])
    assert(len(rows_train) == train_size and len(rows_valid) == valid_size and len(rows_test) == test_size)

    return (rows_train, stats_train, cmds_train), \
        (rows_valid, stats_valid, cmds_valid), \
        (rows_test, stats_test, cmds_test)


def split_by_files(rows, from_file, split, rng):
    dataset_per_file = {}
    for file, stats in from_file.items():
        rows_from_file = [row for row in rows if row[0] in stats]
        if split < 1.0:
            n_rows = len(rows_from_file)
            split_size = int(n_rows * split)
            split_indices = rng.choice(np.arange(n_rows), size=split_size, replace=False)
            rows_from_file = [rows_from_file[i] for i in split_indices]

        dataset_per_file[file] = create_dataset(rows_from_file)
    return dataset_per_file


def create_dataset(rows: List[List[str]]):
    stats = Stats()

    total_commands = set()
    n_commands = []
    for (_, admissible_cmds, cmds) in rows:
        total_commands.update(admissible_cmds)
        n_commands.append(len(admissible_cmds))
        stats.max_cmd_list = max(stats.max_cmd_list, len(cmds))

    # stats
    n_commands = np.array(n_commands)
    stats.states = len(rows)
    stats.unique_cmds = len(total_commands)
    stats.cmds = n_commands.sum()
    stats.mean_cmds_per_state = n_commands.mean()
    stats.std_cmds_per_state = n_commands.std()

    return rows, stats, total_commands


def write_dataset(out_dir, rows, name):
    with open(out_dir / f"{name}.txt", "w") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generates TextWorld ACP dataset.")
    parser.add_argument("logdir", type=str)
    parser.add_argument("--outdir", type=str, default="./data/TextWorldACP")
    parser.add_argument("-s", "--seed", type=int, default=2_183_154_691)

    # split train/valid/test by percentage
    parser.add_argument("--train_split", type=float, default=.85)
    parser.add_argument("--valid_split", type=float, default=.05)

    # options for split by file
    parser.add_argument("--split_by_files", action="store_true", default=False,
                        help="This assumes three files named: train.txt, valid.txt, and test.txt")
    parser.add_argument("--split", type=float, default=1.0, help="How much samples to take from each files. "
                        "Set below 1.0 to create smaller datasets.")
    parser.add_argument("--train_file", type=str, default="train")

    return parser.parse_args()


def main():
    args = parse_args()

    rng = set_rng_seed(args.seed)

    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True, parents=True)

    files = glob(args.logdir + "/**/*.txt", recursive=True)

    if args.split_by_files:
        file_stems = {}
        for file in files:
            file_stems[Path(file).stem] = file
        files = [file_stems["train"], file_stems["valid"], file_stems["test"]]

    rows, total_commands, stats, from_file = parse_files(files)

    stats_str = ""
    stats_str += stats_to_str(stats, "Total", True)
    cmds_train = None
    if not args.split_by_files:
        split = (args.train_split, args.valid_split)
        train, valid, test = split_rows(rows, split, rng)
        rows_train, stats_train, cmds_train = train
        rows_valid, stats_valid, cmds_valid = valid
        rows_test, stats_test, cmds_test = test

        write_dataset(out_dir, rows_train, "train")
        write_dataset(out_dir, rows_valid, "valid")
        write_dataset(out_dir, rows_test, "test")

        stats_str += "\n" + stats_to_str(stats_train, "Train")
        stats_str += "\n" + stats_to_str(stats_valid, "Valid", not_in_train=cmds_valid.difference(cmds_train))
        stats_str += "\n" + stats_to_str(stats_test, "Test", not_in_train=cmds_test.difference(cmds_train))
    else:
        datasets = split_by_files(rows, from_file, args.split, rng)

        if args.train_file in datasets:
            rows_train, stats_train, cmds_train = datasets[args.train_file]
            write_dataset(out_dir, rows_train, args.train_file)
            stats_str += "\n" + stats_to_str(stats_train, args.train_file.capitalize())

        for file, (dataset_rows, dataset_stats, dataset_cmds) in datasets.items():
            if file == args.train_file:
                continue

            write_dataset(out_dir, dataset_rows, file)
            not_in_train = dataset_cmds.difference(cmds_train) if cmds_train is not None else None

            stats_str += "\n" + stats_to_str(dataset_stats, file.capitalize(), not_in_train=not_in_train)

    # write stats
    with open(out_dir / "stats.txt", "w") as fp:
        fp.write(stats_str)

    # write commands
    with open(out_dir / "commands.txt", "w") as fp:
        fp.write("\n".join(total_commands))

    print(stats_str)


if __name__ == "__main__":
    main()
