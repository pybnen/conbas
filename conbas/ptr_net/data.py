from typing import Any, List
from matplotlib.font_manager import json_load
from torch.utils.data.dataset import Dataset
import numpy as np
from torch.utils.data.dataloader import DataLoader
import string
# filter out stop words
from nltk.corpus import stopwords
import torch
from tqdm import tqdm
import random
import numpy as np
import json

PAD_TOKEN = "[PAD]"
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "eos"

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


def process_state(state_arr: List[Any], tokenizer, vocab: Vocabulary):
    ob, admissible_cmds, cmds = state_arr

    ob = preprocess_line(ob, tokenizer, vocab)

    # add EOS to commands
    cmds = [EOS_TOKEN] + [c.strip() for c in cmds]

    admissible_cmds = [c.strip() for c in admissible_cmds]
    filtered_cmds = [c for c in admissible_cmds if c not in ["look", "inventory"]]
    admissible_indices = [cmds.index(c) for c in filtered_cmds]

    cmds = [preprocess_line(c, tokenizer, vocab) for c in cmds]
    return ob, admissible_indices, cmds


class GameStateDataset(Dataset):

    def __init__(self, state_file, tokenizer, vocab: Vocabulary):
        super().__init__()
        self.obs = []
        self.admissible_commands = []
        self.commands_list = []
        self.vocab = vocab

        with open(state_file, "r") as fp:
            lines = fp.readlines()
            with tqdm(lines) as pbar:
                for line in pbar:
                    state_arr = json.loads(line)
                    ob, admissible_cmds, cmds = process_state(state_arr, tokenizer, self.vocab)
                    self.obs.append(ob)
                    self.admissible_commands.append(admissible_cmds)
                    self.commands_list.append(cmds)

    def __getitem__(self, index):
        return self.obs[index], self.admissible_commands[index], self.commands_list[index]

    def __len__(self):
        return len(self.obs)


def create_batch(data):
    state_batch, admissible_cmds_batch, command_list_batch = list(zip(*data))

    # TODO maybe pack
    max_len = max([len(s) for s in state_batch])
    state_padded, state_mask = pad_sequences(state_batch, max_len, 'int')

    max_len = max([len(c) for c in admissible_cmds_batch])
    admissible_cmds_padded, admissible_cmds_mask = pad_sequences(admissible_cmds_batch, max_len, 'int')

    max_len = max([len(c) for command_list in command_list_batch for c in command_list])
    commands = []
    commands_mask = []
    for command_list in command_list_batch:
        command_list_padded, command_list_mask = pad_sequences(command_list, max_len, 'int')
        command_list_padded, command_list_mask = torch.tensor(command_list_padded), torch.tensor(command_list_mask)

        commands.append(command_list_padded)
        commands_mask.append(command_list_mask)

    return torch.tensor(state_padded), torch.tensor(state_mask), \
        torch.tensor(admissible_cmds_padded), torch.tensor(admissible_cmds_mask), \
        commands, commands_mask


def get_dataloader(directory, batch_size, tokenizer, num_workers, seed):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    vocab = Vocabulary()

    g = torch.Generator()
    g.manual_seed(seed)

    ds_train = GameStateDataset(directory + "/train.txt", tokenizer, vocab)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          collate_fn=create_batch, worker_init_fn=seed_worker, generator=g)

    ds_valid = GameStateDataset(directory + "/valid.txt", tokenizer, vocab)
    dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          collate_fn=create_batch, worker_init_fn=seed_worker, generator=g)

    return dl_train, dl_valid, vocab
