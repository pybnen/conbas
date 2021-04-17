from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LstmDqnModel(nn.Module):

    def __init__(self, config: Dict[str, Any], commands: List[str], word_vocab: List[str]) -> None:
        super().__init__()

        self.config = config
        self.embedding = nn.Embedding(len(word_vocab), config["embedding_size"])
        # TODO support bi-directional rnn
        self.representation_rnn = nn.GRU(**config["representation_rnn"])

        linear_layer_hiddens = config["command_scorer_net"] + [len(commands)]
        command_scorer_layers = []
        for i in range(len(config["command_scorer_net"])):
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

    def representation_generator(self, input_tensor: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
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

        hidden = torch.zeros(1, len(input_lengths), self.config["representation_rnn"]["hidden_size"])
        output_packed, hidden = self.representation_rnn(packed, hidden)
        output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=False)

        # https://discuss.pytorch.org/t/average-of-the-gru-lstm-outputs-for-variable-length-sequences/57544/4
        state_representation = output_padded.sum(dim=0) / output_lengths.float().unsqueeze(dim=1)
        return state_representation

    def command_scorer(self, state_representation: torch.Tensor) -> torch.Tensor:
        """Calculates a score for each available command, based on the given state representation.

        Args:
            state_representation: tensor of shape (batch_size, )

        Returns:
            command_scores: tensor fo shape (batch_size, n_commands) for each state a list
                containing the scores of each command in that state
        """
        command_scores = self.command_scorer_net(state_representation)
        return command_scores
