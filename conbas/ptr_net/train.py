from argparse import ArgumentParser
from locale import D_T_FMT
from nltk.tokenize import word_tokenize
import torch
import torch.cuda
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
from data import get_dataloader
from model import Seq2Seq


@torch.no_grad()
def evaluate(seq2seq, data, commands, commands_mask, device, commands_arr):
    seq2seq.eval()
    states, states_masks, admissible_cmds, admissible_masks = data
    states, states_masks = states.to(device), states_masks.to(device)
    admissible_cmds = admissible_cmds.to(device)

    outputs, idxs = seq2seq(states, states_masks, commands, commands_mask)

    label = sorted([commands_arr[i] for i in admissible_cmds.cpu()[:, 0].tolist() if i != 0])
    prediction = sorted([commands_arr[i] for i in idxs.cpu()[:, 0].tolist() if i != 0])
    print("Label:", label)
    print("Pred :", prediction)
    print("------------------------------\n")


def parse_args():
    parser = ArgumentParser(description="Train pointer network")
    parser.add_argument("-bs", "--batch_size", type=int, default=12)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)
    parser.add_argument("dataset_dir", help="directory to dataset")

    return parser.parse_args()


def run():
    args = parse_args()
    # TODO make all variable configs
    lr = 1e-3
    n_epochs = 1_000
    num_workers = args.num_workers

    # get dataloader
    dl_train, dl_valid, vocab, commands_arr \
        = get_dataloader(args.dataset_dir, args.batch_size,
                         tokenizer=word_tokenize, num_workers=num_workers)
    vocab_size = len(vocab)

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get model
    seq2seq = Seq2Seq(device, vocab_size, vocab.word_2_id("[PAD]"))
    seq2seq = seq2seq.to(device)

    # get optimizer
    # TODO params of all stuffs
    optimizer = optim.Adam(seq2seq.parameters(), lr=lr)

    # get loss criterium
    criterion = nn.MSELoss()

    # train
    commands = dl_train.dataset.commands
    commands_mask = dl_train.dataset.commands_mask
    commands, commands_mask = commands.to(device), commands_mask.to(device)
    for epoch in range(n_epochs):
        seq2seq.train()

        running_loss = 0.0
        with tqdm(dl_train) as pbar:
            for data in pbar:
                states, states_masks, admissible_cmds, admissible_cms_mask = data
                states, states_masks = states.to(device), states_masks.to(device)
                admissible_cmds = admissible_cmds.to(device)

                # zero the parameter gradients
                # forward + backward + optimize
                outputs, idxs = seq2seq(states, states_masks, commands, commands_mask)

                admissible_cmd_max_length = admissible_cmds.size(0)
                output_max_lenght = outputs.size(0)
                max_lenght = max(admissible_cmd_max_length, output_max_lenght)

                outputs = F.pad(outputs, (0, 0, 0, 0, 0, max_lenght-output_max_lenght))
                admissible_cmds = F.pad(admissible_cmds, (0, 0, 0, max_lenght-admissible_cmd_max_length))

                outputs_flat = outputs.view(-1, outputs.size(-1))
                admissible_cmds_flatt = admissible_cmds.view(-1)

                loss = F.cross_entropy(outputs_flat, admissible_cmds_flatt)

                # loss = criterion(outputs, admissible_cmds)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        sys.stdout.flush()
        print('Loss: {}'.format(running_loss))
        if epoch % 10 == 0:
            print("Eval:")
            data = iter(dl_train).next()
            evaluate(seq2seq, data, commands, commands_mask, device, commands_arr)

            data = iter(dl_valid).next()
            evaluate(seq2seq, data, commands, commands_mask, device, commands_arr)

    print('Finished Training')


if __name__ == "__main__":
    run()
