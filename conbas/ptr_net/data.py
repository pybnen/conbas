from typing import List
from torch.utils.data.dataset import Dataset
import numpy as np
from torch.utils.data.dataloader import DataLoader
import string
# filter out stop words
from nltk.corpus import stopwords
import torch
from tqdm import tqdm
import csv

PAD_TOKEN = "[PAD]"
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"

PAD_TOKEN_ID = 0
SOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2

table = str.maketrans('', '', string.punctuation)
stop_words = set(stopwords.words('english'))


def pad_sequences(data, pad_length, dtype):
    padded_data = np.zeros((len(data), pad_length), dtype=dtype)
    padded_mask = np.zeros((len(data), pad_length), dtype='int8')

    for i, line in enumerate(data):
        if len(line) > pad_length:
            line = line[len(line) - pad_length:]
        padded_data[i, :len(line)] = line
        padded_mask[i, :len(line)] = 1
    return padded_data, padded_mask


class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_TOKEN_ID: PAD_TOKEN,
                           SOS_TOKEN_ID: SOS_TOKEN,
                           EOS_TOKEN_ID: EOS_TOKEN
                           }
        for id, w in self.index2word.items():
            self.word2index[w] = id
            self.word2count[w] = 0

        self.n_words = len(self.index2word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def word_2_id(self, word):
        self.add_word(word)
        return self.word2index[word]

    def id_2_word(self, id):
        return self.index2word[id]

    def to_sentence(self, ids: List[int]):
        return " ".join([self.id_2_word(id) for id in ids])

    def __len__(self):
        assert (len(self.index2word) == len(self.word2index))
        return len(self.word2index)


def preprocess_line(line, tokenizer, vocab: Vocabulary, remove_stopwords=True):
    # remove spaces
    # TODO check if: re.sub("\s+", " ", line) is cheaper
    # TODO check if: while "" in words: words.remove("") is cheaper
    words = line.split()
    words = [w for w in words if len(w) > 0]
    line = " ".join(words)

    # tokenizer
    tokens = tokenizer(line)

    tokens = [w.lower() for w in tokens]

    # remove stop words
    tokens = [w for w in tokens if w not in stop_words]

    # remove punctuation from each word
    tokens = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic
    tokens = [w for w in tokens if w.isalpha()]

    # think i don't need sos/eos because this is not a
    # seq2seq task, instead a feature vector is generated
    # tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]
    return [vocab.word_2_id(w) for w in tokens]


def parse_row(row: List[str], commands: List[str], tokenizer, vocab: Vocabulary):
    state_line, admissible_cmds = row[0], row[1:]
    state = preprocess_line(state_line, tokenizer, vocab)

    admissible_cmds = [cmd.strip() for cmd in admissible_cmds]
    filtered_cmds = [cmd for cmd in admissible_cmds
                     if cmd not in ["look", "inventory"]]

    # will throw exception of admissible commands not found -> good
    admissible_idx = [commands.index(cmd) for cmd in filtered_cmds]
    return state, admissible_idx


class GameStateDataset(Dataset):

    def __init__(self, state_file, commands_file, tokenizer, vocab: Vocabulary):
        super().__init__()
        self.states = []
        self.admissible_commands = []
        self.vocab = vocab

        with open(commands_file, "r") as fp:
            self.commands_arr = [EOS_TOKEN] + [line.strip() for line in fp.readlines()]
            self.commands = [preprocess_line(cmd, tokenizer, self.vocab) for cmd in self.commands_arr]
            max_len = max([len(cmd) for cmd in self.commands])
            padded_cmds, cmds_mask = pad_sequences(self.commands, max_len, 'int')
            self.commands = torch.tensor(padded_cmds).T
            self.commands_mask = torch.tensor(cmds_mask).T

        with open(state_file, "r") as fp:
            lines = fp.readlines()
            reader = csv.reader(lines)
            with tqdm(reader) as pbar:
                for row in pbar:
                    state, admissible_cmds = parse_row(row, self.commands_arr, tokenizer, self.vocab)
                    self.states.append(state)
                    self.admissible_commands.append(admissible_cmds)

    def __getitem__(self, index):
        return self.states[index], self.admissible_commands[index]

    def __len__(self):
        return len(self.states)


def create_batch(data):
    state_batch, admissible_cmds_batch = list(zip(*data))

    # TODO maybe pack
    max_len = max([len(s) for s in state_batch])
    padded_state_batch, state_mask_batch = pad_sequences(state_batch, max_len, 'int')

    max_len = max([len(c) for c in admissible_cmds_batch])
    padded_admissible_cmds, admissible_cmds_mask_batch = pad_sequences(admissible_cmds_batch, max_len, 'int')

    return torch.tensor(padded_state_batch).T, \
        torch.tensor(state_mask_batch).T, \
        torch.tensor(padded_admissible_cmds).T, \
        torch.tensor(admissible_cmds_mask_batch).T


def get_dataloader(directory, batch_size, tokenizer, num_workers):
    vocab = Vocabulary()

    ds_train = GameStateDataset(directory + "/train.txt", directory + "/commands.txt", tokenizer, vocab)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          collate_fn=create_batch)

    ds_valid = GameStateDataset(directory + "/valid.txt", directory + "/commands.txt", tokenizer, vocab)
    dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          collate_fn=create_batch)

    return dl_train, dl_valid, vocab, ds_train.commands_arr
