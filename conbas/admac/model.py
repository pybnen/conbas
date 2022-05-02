import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
#  from data import SOS_TOKEN_ID, EOS_TOKEN_ID


class ADMAC(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_hidden_size, classifier_hidden_sizes, padding_idx):
        super().__init__()

        self.state_embedding = StateEmbedding(vocab_size, embedding_size, rnn_hidden_size, padding_idx)
        state_embedding_size = rnn_hidden_size * 2

        linear_layer_hiddens = [state_embedding_size] + classifier_hidden_sizes + [1]

        layers = []
        for i in range(len(linear_layer_hiddens) - 1):
            input_size = linear_layer_hiddens[i]
            output_size = linear_layer_hiddens[i+1]
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
        layers = layers[:-1]
        # layers.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*layers)

    def forward(self, state, state_mask, cmd, cmd_mask):
        # n_batch, hidden * 2
        state_embed = self.state_embedding(state, state_mask, cmd, cmd_mask)

        # n_batch, 1
        pred = self.classifier(state_embed)
        return pred


class StateEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=hidden_size,
                           num_layers=1,
                           bidirectional=False)

    def representation(self, input, input_mask):
        # n_batch, n_seq, hidden
        embedded_input = self.embedding(input)
        # n_seq, n_batch, hidden
        embedded_input = embedded_input.permute((1, 0, 2))

        output, (h, c) = self.rnn(embedded_input)

        # mean pooling, but take padding into account
        input_mask = input_mask.permute((1, 0)).unsqueeze(-1)
        masked_output = output * input_mask
        repr = masked_output.sum(dim=0) / input_mask.sum(dim=0)

        # n_batch, hidden
        return repr

    def forward(self, state, state_mask, cmd, cmd_mask):
        # n_batch, hidden
        state_repr = self.representation(state, state_mask)
        # n_batch, hidden
        cmd_repr = self.representation(cmd, cmd_mask)

        # n_batch, n_cmds, hidden * 2
        output = torch.cat([state_repr, cmd_repr], dim=1)

        return output
