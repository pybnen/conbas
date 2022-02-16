from argparse import ArgumentParser
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

    print("Eval:")

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
    commands_arr = ["[EOS]",
                    "go north", "go east", "go south", "go west",
                    "examine counter", "examine lettuce", "examine blue door", "examine blue key", "examine tomato",
                    "take lettuce", "take blue key", "take tomato",
                    "take lettuce from counter", "take blue key from counter", "take tomato from counter",
                    "drop lettuce", "drop blue key", "drop tomato",
                    "eat lettuce", "eat tomato",
                    "close blue door",
                    "open blue door",
                    "unlock blue door with blue key",
                    "put lettuce on counter", "put blue key on counter", "put tomato on counter",
                    "lock blue door with blue key"]

    # get dataloader
    dl = get_dataloader(args.dataset_dir, commands_arr, args.batch_size,
                        tokenizer=word_tokenize, num_workers=num_workers)
    vocab_size = len(dl.dataset.vocab)

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get model
    seq2seq = Seq2Seq(device, vocab_size, dl.dataset.vocab.word_2_id("[PAD]"))
    seq2seq = seq2seq.to(device)

    # get optimizer
    # TODO params of all stuffs
    optimizer = optim.Adam(seq2seq.parameters(), lr=lr)

    # get loss criterium
    criterion = nn.MSELoss()

    # train
    commands = dl.dataset.commands
    commands_mask = dl.dataset.commands_mask
    commands, commands_mask = commands.to(device), commands_mask.to(device)
    for epoch in range(n_epochs):
        seq2seq.train()

        running_loss = 0.0
        # dl_iter = iter(dl)
        with tqdm(dl) as pbar:
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
            data = iter(dl).next()
            evaluate(seq2seq, data, commands, commands_mask, device, commands_arr)

    print('Finished Training')


if __name__ == "__main__":
    run()
