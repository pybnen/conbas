from data import Vocabulary, GameStateDataset
from argparse import ArgumentParser
from nltk.tokenize import word_tokenize
from pathlib import Path


def parse_args():
    parser = ArgumentParser(description="Train admissible action classifier")
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("-s", "--seed", type=int, default=2_183_154_691)
    return parser.parse_args()


def main():
    args = parse_args()

    vocab = Vocabulary()
    GameStateDataset(args.datadir + "/train.txt", word_tokenize, vocab)
    GameStateDataset(args.datadir + "/valid.txt", word_tokenize, vocab)

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    with open(logdir / "vocab.txt", "w") as fp:
        fp.write(vocab.serialize())


if __name__ == '__main__':
    main()
