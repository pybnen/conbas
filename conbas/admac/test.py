from argparse import ArgumentParser
from unittest.mock import patch
from nltk.tokenize import word_tokenize
import torch
import torch.cuda
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
import random
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import yaml
from functools import reduce
import shutil
import sklearn.metrics
import csv
import datetime

from data import get_testset_loader, Vocabulary
from model import ADMAC
from logger import Logger

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

torch.backends.cudnn.benchmark = False
# try:
#     torch.use_deterministic_algorithms(True)
# except AttributeError as e:
#     print(e)

def set_rng_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return np.random.default_rng(seed)


def parse_args():
    parser = ArgumentParser(description="Test admissible action classifier")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-s", "--seed", type=int, default=2_183_154_691)
    return parser.parse_args()


def calc_metrics(probas_pred, label):
    pred = probas_pred.round()
    accuracy = sklearn.metrics.accuracy_score(label, pred)
    f1 = sklearn.metrics.f1_score(label, pred, zero_division=0)
    precision = sklearn.metrics.precision_score(label, pred, zero_division=0)
    recall = sklearn.metrics.recall_score(label, pred, zero_division=0)

    return accuracy, f1, precision, recall


def main():
    args = parse_args()

    with open(args.config, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)
    config['args'] = vars(args)

    _ = set_rng_seed(args.seed)

    # logger
    logdir = Path(config['logdir'])
    logdir.mkdir(parents=True, exist_ok=True)
    logger = Logger(None)

    # get dataloader
    print("Load dataset")
    vocab = Vocabulary.deserialize(logdir / config['vocab'])
    dl_test = \
        get_testset_loader(config['test']['datadir'],
                           vocab,
                           config['test']['batch_size'],
                           tokenizer=word_tokenize,
                           num_workers=config['test']['num_workers'],
                           seed=args.seed)
    vocab_size = len(vocab)

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get model
    embedding_size = config['model']['embedding_size']
    rnn_hidden_size = config['model']['rnn_hidden_size']
    classifier_hidden_sizes = config['model']['classifier_hidden_sizes']

    model = ADMAC(vocab_size, embedding_size, rnn_hidden_size, classifier_hidden_sizes, vocab.word_2_id("[PAD]"))
    model = model.to(device)

    state_dict = torch.load(logdir / config['state_dict'], map_location=device)
    model.load_state_dict(state_dict)

    print("\nStart test")
    model.eval()

    with tqdm(dl_test) as pbar:
        for data in pbar:
            # n_batch, n_seq
            states, states_masks = data[0].to(device), data[1].to(device)
            # n_batch, n_seq
            cmd, cmd_mask = data[2].to(device), data[3].to(device)
            # n_batch
            admissible = data[4].to(device)

            with torch.no_grad():
                logits = model(states, states_masks, cmd, cmd_mask)

                loss = F.binary_cross_entropy_with_logits(logits, admissible.reshape(-1, 1))

            # log statistics
            probas_pred = torch.sigmoid(logits.detach()).reshape(-1).cpu().numpy()
            label = admissible.cpu().numpy()
            batch_size = states.size(0)

            metrics = calc_metrics(probas_pred, label)
            logger.add_step(batch_size, loss.item(), *metrics)
    logger.end_epoch()
    sys.stdout.flush()

    # print statistics
    print("  test     :", logger.to_str())

    with open(logdir / "test.txt", "a") as fp:
        fp.write("{} {} {}".format(datetime.datetime.now(), config['state_dict'], logger.to_str() + "\n"))


if __name__ == '__main__':
    main()
