from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from ..helpers.layers import Embedding, masked_mean, FastUniLSTM


class LstmDrqnModel(nn.Module):

    def __init__(self, config: Dict[str, Any], commands: List[str], word_vocab: List[str],
                 device: torch.device) -> None:
        super().__init__()

        self.config = config
        self.device = device

        self.embedding_size = config["embedding_size"]
        self.representation_hidden_size = config['representation_rnn']
        self.command_scorer_rnn_hidden = config['command_scorer_rnn_hidden']

        # define representation rnn
        if config["type"] == "lstm":
            self.embedding = nn.Embedding(len(word_vocab), self.embedding_size)
            self.representation_rnn = nn.LSTM(input_size=self.embedding_size,
                                              hidden_size=self.representation_hidden_size[0],
                                              num_layers=1)
            self.representation_generator = self._representation_generator
        elif config["type"] == "gru":
            self.embedding = nn.Embedding(len(word_vocab), self.embedding_size)
            self.representation_rnn = nn.GRU(input_size=self.embedding_size,
                                             hidden_size=self.representation_hidden_size[0],
                                             num_layers=1)
            self.representation_generator = self._representation_generator
        elif config["type"] == "fast_lstm":
            # define layers
            self.embedding = Embedding(embedding_size=self.embedding_size,
                                       vocab_size=len(word_vocab),
                                       enable_cuda=device.type == "cuda")

            self.representation_rnn = FastUniLSTM(ninp=self.embedding_size,
                                                  nhids=self.representation_hidden_size,
                                                  dropout_between_rnn_layers=0.)
            self.representation_generator = self._representation_generator_fast
        else:
            raise ValueError

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

    def _representation_generator(self, input_tensor: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        """Generates a state representation based on input observation.

        Input is a padded list of state information (sequences), an RNN is used to create features of the sequences.
        The state representation is the mean of these features.

        Args:
            input_tensor: tensor of shape (max_len, batch_size) containing the padded state information
                for each game in the batch.
                max_len: maximum length of game information
                batch_size: number of games in batch
            input_lengths: tensor of shape (batch_size) containing the length of each input sequence

        Returns:
            state_representation: tensor of shape (batch_size, )
        """
        embed = self.embedding(input_tensor)
        packed = pack_padded_sequence(embed, input_lengths, enforce_sorted=False)

        output_packed, hidden = self.representation_rnn(packed)
        output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=False)

        # https://discuss.pytorch.org/t/average-of-the-gru-lstm-outputs-for-variable-length-sequences/57544/4
        output_lengths_device = output_lengths.float().to(self.device)
        state_representation = output_padded.sum(dim=0) / output_lengths_device.unsqueeze(dim=1)
        return state_representation

    def _representation_generator_fast(self, input_tensor: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        """Generates a state representation based on input observation.

        Input is a padded list of state information (sequences), an RNN is used to create features of the sequences.
        The state representation is the mean of these features.

        Args:
            input_tensor: tensor of shape (max_len, batch_size) containing the padded state information
                for each game in the batch.
                max_len: maximum length of game information
                batch_size: number of games in batch
            input_lengths: tensor of shape (batch_size) containing the length of each input sequence

        Returns:
            state_representation: tensor of shape (batch_size, )
        """
        input_tensor = input_tensor.transpose(0, 1)  # fast lstm impl expects batch first
        embed, mask = self.embedding.forward(input_tensor)
        encoding_sequence, _, _ = self.representation_rnn.forward(embed, mask)
        mean_encoding = masked_mean(encoding_sequence, mask)  # batch x h
        return mean_encoding

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
