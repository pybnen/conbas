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

from data import get_dataloader
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
    parser = ArgumentParser(description="Train admissible action classifier")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-o", "--overwrite", action="store_true", default=False, help="Overwrite log file")
    parser.add_argument("-s", "--seed", type=int, default=2_183_154_691)
    return parser.parse_args()


def calc_metrics(probas_pred, label):
    pred = probas_pred.round()
    accuracy = sklearn.metrics.accuracy_score(label, pred)
    f1 = sklearn.metrics.f1_score(label, pred, zero_division=0)
    precision = sklearn.metrics.precision_score(label, pred, zero_division=0)
    recall = sklearn.metrics.recall_score(label, pred, zero_division=0)

    return accuracy, f1, precision, recall


@torch.no_grad()
def get_example(model, dl, device):
    # get random batch from dataloader
    dl_iter = iter(dl)
    dl_len = len(dl)
    data = next(dl_iter)
    for _ in range(np.random.randint(dl_len)):
        data = next(dl_iter)

     # n_batch, n_seq
    states, states_masks = data[0].to(device), data[1].to(device)
    # n_batch, n_seq
    cmd, cmd_mask = data[2].to(device), data[3].to(device)
    # n_batch
    admissible = data[4].to(device)

    logits = model(states, states_masks, cmd, cmd_mask)

    batch_size = states.size(0)
    pred = torch.sigmoid(logits.detach()).reshape(-1).cpu().numpy().round()
    label = admissible.cpu().numpy()

    rows = []
    for i in range(batch_size):
        str_ob = dl.dataset.vocab.to_sentence(states.cpu()[i, states_masks[i, :].bool()].tolist())
        str_cmd = dl.dataset.vocab.to_sentence(cmd.cpu()[i, cmd_mask[i, :].bool()].tolist())
        rows.append((str_ob, str_cmd, int(pred[i]), int(label[i])))

    return rows


@torch.no_grad()
def evaluate(model, dl_valid, device, logger):
    model.eval()

    logger.reset()
    for data in dl_valid:
        # n_batch, n_seq
        states, states_masks = data[0].to(device), data[1].to(device)
        # n_batch, n_seq
        cmd, cmd_mask = data[2].to(device), data[3].to(device)
        # n_batch
        admissible = data[4].to(device)

        logits = model(states, states_masks, cmd, cmd_mask)

        loss = F.binary_cross_entropy_with_logits(logits, admissible.reshape(-1, 1))

        # log statistics
        probas_pred = torch.sigmoid(logits.detach()).reshape(-1).cpu().numpy()
        label = admissible.cpu().numpy()
        batch_size = states.size(0)

        metrics = calc_metrics(probas_pred, label)
        logger.add_step(batch_size, loss.item(), *metrics)

    logger.end_epoch()


def main():
    args = parse_args()

    with open(args.config, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)
    config['args'] = vars(args)

    _ = set_rng_seed(args.seed)

    lr = config['training']['lr']
    n_epochs = config['training']['epochs']

    # logger
    logdir = Path(config['logdir'])
    if logdir.exists():
        if args.overwrite:
            print(f"Overwrite logdir '{config['logdir']}'")
            shutil.rmtree(logdir)
        else:
            print(f"Logdir already exists: '{config['logdir']}'")
            return

    logdir.mkdir(parents=True, exist_ok=False)

    # copy config file
    with open(logdir / "config.yaml", "w") as fp:
        fp.write(yaml.dump(config))

    writer = SummaryWriter(log_dir=logdir)
    logger = Logger(writer)
    logger_val = Logger(writer)

    # get dataloader
    print("Load dataset")
    dl_train, dl_valid, vocab = \
        get_dataloader(config['training']['datadir'],
                       config['training']['batch_size'],
                       tokenizer=word_tokenize,
                       num_workers=config['training']['num_workers'],
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

    # get optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # learning rate scheduler
    scheduler_config = config['training']['scheduler']
    scheduler = None
    if scheduler_config:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=scheduler_config['factor'],
                                                         patience=scheduler_config['patience'],
                                                         verbose=True)

    print("\nStart training")
    train_step = 0
    best_val_loss = float('inf')
    best_val_accuracy = float('-inf')
    for epoch in range(n_epochs):
        # train
        model.train()
        logger.reset()
        with tqdm(dl_train) as pbar:
            for data in pbar:
                # n_batch, n_seq
                states, states_masks = data[0].to(device), data[1].to(device)
                # n_batch, n_seq
                cmd, cmd_mask = data[2].to(device), data[3].to(device)
                # n_batch
                admissible = data[4].to(device)

                logits = model(states, states_masks, cmd, cmd_mask)

                loss = F.binary_cross_entropy_with_logits(logits, admissible.reshape(-1, 1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # log statistics
                probas_pred = torch.sigmoid(logits.detach()).reshape(-1).cpu().numpy()
                label = admissible.cpu().numpy()
                batch_size = states.size(0)

                metrics = calc_metrics(probas_pred, label)
                logger.add_step(batch_size, loss.item(), *metrics)

                train_step += batch_size
        logger.end_epoch()
        sys.stdout.flush()

        # validate
        evaluate(model, dl_valid, device, logger_val)
        current_val_loss = logger_val.epoch_loss
        current_val_accuracy = logger_val.epoch_accuracy
        mv_avg_val_loss = logger_val.mv_avg_epoch_loss

        if scheduler:
            if scheduler_config['target'] == 'current_loss':
                scheduler.step(current_val_loss)
            elif scheduler_config['target'] == 'mv_avg_loss':
                scheduler.step(mv_avg_val_loss)
            else:
                raise ValueError

        str_best = ""
        if best_val_loss > current_val_loss:
            best_val_loss = current_val_loss
            str_best = "*new best loss* "
            # save network
            torch.save(model.state_dict(), logdir / "best_loss_state_dict.pth")

        if best_val_accuracy < current_val_accuracy:
            best_val_accuracy = current_val_accuracy
            str_best += "*new best accuracy*"
            # save network
            torch.save(model.state_dict(), logdir / "best_accuracy_state_dict.pth")

        # log examples
        if epoch % config['log_example_interval'] == 0:
            rows = get_example(model, dl_valid, device)
            with open(logdir / "examples.txt", "a") as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerows(rows)

        # log statistics
        print("  train     :", logger.to_str())
        print("  validation:", logger_val.to_str(), str_best)
        logger.log("train", train_step)
        logger_val.log("valid", train_step)

    print('Finished Training')


if __name__ == '__main__':
    main()
