from sre_parse import State
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AdmacDrqn(nn.Module):
    def __init__(self, config: Dict[str, Any], commands: List[str], word_vocab: List[str],
                 device: torch.device) -> None:
        super().__init__()

        self.config = config
        self.device = device

        self.embedding_size = config["embedding_size"]
        self.representation_hidden_size = config['representation_rnn']
        self.command_scorer_rnn_hidden = config['command_scorer_rnn_hidden']

        self.embedding = Embedding(len(word_vocab), self.embedding_size, self.representation_hidden_size[0], 0)

        # define command scorer rnn
        self.command_scorer_rnn = nn.LSTMCell(input_size=self.representation_hidden_size[-1],
                                              hidden_size=self.command_scorer_rnn_hidden,
                                              bias=True)

        linear_layer_hiddens = [self.command_scorer_rnn_hidden] +  \
            config["command_scorer_linear"] + [len(commands)]

        command_scorer_layers = []
        for i in range(len(linear_layer_hiddens) - 1):
            input_size = linear_layer_hiddens[i]
            output_size = linear_layer_hiddens[i+1]
            command_scorer_layers.append(nn.Linear(input_size, output_size))
            command_scorer_layers.append(nn.ReLU())
        self.command_scorer_net = nn.Sequential(*command_scorer_layers[:-1])  # skip last relu layer
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights of the model"""
        # TODO actually not quite sure how to initialize GRU

        for layer in self.command_scorer_net:
            if type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight.data)  # type: ignore
                layer.bias.data.fill_(0)  # type: ignore

    def command_scorer(self, state_representation: torch.Tensor,
                       h_0: torch.Tensor = None,
                       c_0: torch.Tensor = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Calculates a score for each available command, based on the given state representation.

        Args:
            state_representation: tensor of shape (batch_size, )

        Returns:
            command_scores: tensor fo shape (batch_size, n_commands) for each state a list
                containing the scores of each command in that state
        """
        # h_0, c_0 either both are set or both are none
        assert (h_0 is None) == (c_0 is None)
        hidden = None if h_0 is None else (h_0, c_0)

        h_1, c_1 = self.command_scorer_rnn(state_representation, hidden)
        command_scores = self.command_scorer_net(h_1)
        return command_scores, (h_1, c_1)


class ADMAC(nn.Module):
    def __init__(self, state_size, hidden_sizes):
        super().__init__()

        linear_layer_hiddens = [state_size * 2] + hidden_sizes + [1]
        layers = []
        for i in range(len(linear_layer_hiddens) - 1):
            state_size = linear_layer_hiddens[i]
            output_size = linear_layer_hiddens[i+1]
            layers.append(nn.Linear(state_size, output_size))
            layers.append(nn.ReLU())
        # remove last relu layer
        layers = layers[:-1]
        self.classifier = nn.Sequential(*layers)

    def forward(self, state_embed, cmd_embed):
        state_cmd_embed = torch.cat([state_embed, cmd_embed], dim=-1)
        return self.classifier(state_cmd_embed)


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=hidden_size,
                           num_layers=1,
                           bidirectional=False)

    def forward(self, input: torch.Tensor, length: torch.Tensor):
        embed = self.embedding(input)
        packed = pack_padded_sequence(embed, length, enforce_sorted=False)

        output_packed, hidden = self.rnn(packed)
        output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=False)

        # https://discuss.pytorch.org/t/average-of-the-gru-lstm-outputs-for-variable-length-sequences/57544/4
        output_lengths_device = output_lengths.float()
        repr = output_padded.sum(dim=0) / output_lengths_device.unsqueeze(dim=1)
        return repr
