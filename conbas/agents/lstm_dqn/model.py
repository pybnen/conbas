from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LstmDqnModel(nn.Module):

    def __init__(self, config: Dict[str, Any], commands: List[str], word_vocab: List[str]) -> None:
        super().__init__()

        self.config = config
        # convert the word id to an embedding
        self.embedding = nn.Embedding(len(word_vocab), config["embedding_size"])
        self.representation_rnn = nn.GRU(config["embedding_size"], config["hidden_size"])
        # TODO maybe add relu
        self.command_scorer_linear = nn.Linear(config["hidden_size"], len(commands))
        self.init_weights()

    def init_weights(self):
        # TODO actually not quite sure how to initialize GRU
        nn.init.xavier_uniform_(self.command_scorer_linear.weight.data)
        self.command_scorer_linear.bias.data.fill_(0)

    def representation_generator(self, game_step_info: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
        """TODO

        Args:
            game_step_info: tensor of size (max_len, batch_size)
            sequence_lengths: list of sequences lengths of each batch element

        Returns:
            TODO
        """
        embed = self.embedding(game_step_info)
        packed = pack_padded_sequence(embed, sequence_lengths, enforce_sorted=False)

        hidden = torch.zeros(1, len(sequence_lengths), self.config["hidden_size"])
        output_packed, hidden = self.representation_rnn(packed, hidden)
        output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=False)

        # https://discuss.pytorch.org/t/average-of-the-gru-lstm-outputs-for-variable-length-sequences/57544/4
        state_representation = output_padded.sum(dim=0) / output_lengths.float().unsqueeze(dim=1)
        return state_representation

    def command_scorer(self, state_representation):
        """TODO
        
        Args:
            state_representation: TODO

        Returns:
            TODO
        """
        output = self.command_scorer_linear(state_representation)
        return output
