from argparse import ArgumentParser
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

from data import get_dataloader
from model import Seq2Seq
from logger import Logger
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

torch.backends.cudnn.benchmark = False
# try:
#     torch.use_deterministic_algorithms(True)
# except AttributeError as e:
#     print(e)


@torch.no_grad()
def evaluate(seq2seq, dl_valid, device, epoch, logger):
    seq2seq.eval()

    logger.reset()
    for data in dl_valid:
        states, states_masks = data[0].to(device), data[1].to(device)
        admissible_cmds = data[2].to(device)
        # admissible_cms_mask = data[3].to(device)
        # commands, commands_mask = data[4].to(device), data[5].to(device)
        commands = [c.to(device) for c in data[4]]
        commands_mask = [c.to(device) for c in data[5]]

        batch_size = states.size(0)

        outputs, indices = seq2seq(states, states_masks, commands, commands_mask)

        admissible_cmd_max_length = admissible_cmds.size(1)
        output_max_lenght = outputs.size(1)
        max_lenght = max(admissible_cmd_max_length, output_max_lenght)

        outputs = F.pad(outputs, (0, 0, 0, max_lenght-output_max_lenght, 0, 0))
        admissible_cmds = F.pad(admissible_cmds, (0, max_lenght-admissible_cmd_max_length, 0, 0))

        outputs_flat = outputs.view(-1, outputs.size(-1))
        admissible_cmds_flatt = admissible_cmds.view(-1)

        loss = F.cross_entropy(outputs_flat, admissible_cmds_flatt)

        # log statistics
        admissible_cmds_flatt = admissible_cmds_flatt.detach().cpu()

        indices_flatt = F.pad(indices, (0, max_lenght-output_max_lenght, 0, 0))
        indices_flatt = indices_flatt.detach().cpu().reshape(-1)

        mask = ((indices_flatt + admissible_cmds_flatt) > 0).float()
        accuracy = torch.sum((indices_flatt == admissible_cmds_flatt).float() * mask) / mask.sum()

        logger.add_step(loss.item(), accuracy.item(), batch_size)

    logger.end_epoch()

    if epoch % 10 == 0:
        ob = dl_valid.dataset.vocab.to_sentence(states.cpu()[0, states_masks[0, :].bool()].tolist())
        commands_arr = [dl_valid.dataset.vocab.to_sentence(
            c[cm.bool()].tolist()) for c, cm in zip(commands[0], commands_mask[0])]

        # commands_arr = commands_arr + ["OOI"] * (1 + max(idxs.cpu()[0, :].tolist()) - len(commands_arr))

        label = sorted([commands_arr[i] for i in admissible_cmds.cpu()[0, :].tolist() if i != 0])
        prediction = sorted([commands_arr[i] for i in indices.cpu()[0, :].tolist() if i != 0])
        print("\n\nObservation", ob)
        print("------------------------------\n")
        print("Label:", label)
        print("------------------------------\n")
        print("Pred :", prediction)
        print("------------------------------\n")


def set_rng_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return np.random.default_rng(seed)


def parse_args():
    parser = ArgumentParser(description="Train pointer network")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-s", "--seed", type=int, default=2_183_154_691)
    return parser.parse_args()


def run():
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
        print(f"Logdir already exists: '{config['logdir']}'")
        return
    logdir.mkdir(parents=True)

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
    hidden_size = config['model']['hidden_size']
    seq2seq = Seq2Seq(vocab_size, embedding_size, hidden_size, device, vocab.word_2_id("[PAD]"))
    seq2seq = seq2seq.to(device)

    # get optimizer
    optimizer = optim.Adam(seq2seq.parameters(), lr=lr)

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
    best_validation_loss = float('inf')
    for epoch in range(n_epochs):
        # train
        seq2seq.train()
        logger.reset()
        with tqdm(dl_train) as pbar:
            for data in pbar:
                # n_batch, n_seq
                states, states_masks = data[0].to(device), data[1].to(device)
                # n_batch, n_admissible_cmds
                admissible_cmds = data[2].to(device)  # admissible_cms_mask = data[3].to(device)
                # List[n_cmds[i], cmd_len] for i = 0 ... n_batch -1
                commands = [c.to(device) for c in data[4]]
                commands_mask = [c.to(device) for c in data[5]]

                outputs, indices = seq2seq(states, states_masks, commands, commands_mask)

                admissible_cmd_max_length = admissible_cmds.size(1)
                output_max_lenght = outputs.size(1)
                max_lenght = max(admissible_cmd_max_length, output_max_lenght)

                outputs = F.pad(outputs, (0, 0, 0, max_lenght-output_max_lenght, 0, 0))
                admissible_cmds = F.pad(admissible_cmds, (0, max_lenght-admissible_cmd_max_length, 0, 0))

                outputs_flat = outputs.view(-1, outputs.size(-1))
                admissible_cmds_flatt = admissible_cmds.view(-1)

                # TODO mask out padding
                loss = F.cross_entropy(outputs_flat, admissible_cmds_flatt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # log statistics
                admissible_cmds_flatt = admissible_cmds_flatt.detach().cpu()

                indices_flatt = F.pad(indices, (0, max_lenght-output_max_lenght, 0, 0))
                indices_flatt = indices_flatt.detach().cpu().reshape(-1)

                mask = ((indices_flatt + admissible_cmds_flatt) > 0).float()
                accuracy = torch.sum((indices_flatt == admissible_cmds_flatt).float() * mask) / mask.sum()

                batch_size = states.size(0)
                logger.add_step(loss.item(), accuracy.item(), batch_size)

                train_step += batch_size
        logger.end_epoch()
        sys.stdout.flush()

        # validate
        evaluate(seq2seq, dl_valid, device, epoch, logger_val)
        current_val_loss = logger_val.epoch_loss
        mv_avg_val_loss = logger_val.mv_avg_epoch_loss

        if scheduler:
            if scheduler_config['target'] == 'current_loss':
                scheduler.step(current_val_loss)
            elif scheduler_config['target'] == 'mv_avg_loss':
                scheduler.step(mv_avg_val_loss)
            else:
                raise ValueError

        new_best = False
        if best_validation_loss > current_val_loss:
            best_validation_loss = current_val_loss
            new_best = True
            # save network
            torch.save(seq2seq.state_dict(), logdir / "state_dict.pth")

        # log statistics
        print("  train     :", logger.to_str())
        print("  validation:", logger_val.to_str(), "*new best*" if new_best else "")
        logger.log("train", train_step)
        logger_val.log("valid", train_step)

    print('Finished Training')


if __name__ == "__main__":
    run()