import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
#  from data import SOS_TOKEN_ID, EOS_TOKEN_ID

EMBEDDING_SIZE = 64
HIDDEN_SIZE = 64


class Seq2Seq(nn.Module):
    def __init__(self, device, vocab_size, padding_idx):
        super().__init__()
        self.state_embedding = StateEmbedding(vocab_size, padding_idx)
        self.encoder = Encoder()
        self.seq_decoder = SeqDecoder(device)

    def forward(self, state, state_mask, cmds, cmds_mask):
        # [n_cmds, n_batch, hidden * 2]
        state_embed = self.state_embedding(state, state_mask, cmds, cmds_mask)

        encoder_outputs, hidden_states = self.encoder(state_embed)

        outputs, idxs = self.seq_decoder(encoder_outputs, hidden_states)

        return outputs, idxs


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.LSTM(input_size=HIDDEN_SIZE * 2,
                           hidden_size=HIDDEN_SIZE * 2,
                           num_layers=1,
                           bidirectional=False)

    def forward(self, input: torch.Tensor):
        output, (hidden, cell) = self.rnn(input)
        return output, (hidden, cell)


class SeqDecoder(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.decoder = Decoder()
        self.v = nn.Linear(HIDDEN_SIZE * 2, 1, bias=False)
        self.W1 = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE * 2, bias=False)
        self.W2 = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE * 2, bias=False)

        self.device = device

    def forward(self, encoder_outputs, hidden_state):
        assert torch.all(encoder_outputs[-1] == hidden_state[0])
        seq_len, n_batch, n_hidden = encoder_outputs.size()

        # SOS
        input = torch.zeros_like(encoder_outputs[0])  # torch.zeros(n_batch, hidden_size).to(self.device)

        outputs = []
        idxs = []

        max_length = seq_len
        # while output != "EOS":
        #  while pointer_idx != eos_idx:
        for Ö in range(max_length):
            output_att, hidden_state = self.decoder(input, hidden_state, encoder_outputs)
            dist = Categorical(logits=output_att)
            input_idx = dist.sample()

            index_tensor = input_idx.unsqueeze(0).unsqueeze(-1).expand(1, n_batch, n_hidden)
            input = encoder_outputs.gather(dim=0, index=index_tensor).squeeze(0)
            # use gather instead, like:
            # decoder_input = torch.gather(encoder_outputs, dim=1, index=index_tensor).squeeze(1)1
            input2 = []
            for i, idx in enumerate(input_idx):
                input2.append(encoder_outputs[idx, i, :])
            input2 = torch.stack(input2).detach()

            assert(torch.all(input2 == input))

            idxs.append(input_idx)
            outputs.append(output_att)

        outputs = torch.stack(outputs)
        idxs = torch.stack(idxs)
        return outputs, idxs


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.rnn = nn.LSTM(input_size=HIDDEN_SIZE * 2,
                           hidden_size=HIDDEN_SIZE * 2,
                           num_layers=1,
                           bidirectional=False)
        self.att = Attention()

    def forward(self, input: torch.Tensor, hidden_state: torch.Tensor, encoder_outputs: torch.Tensor):
        hidden, cell = hidden_state

        output, (hidden, cell) = self.rnn(input.unsqueeze(dim=0), (hidden, cell))
        output = output.squeeze(dim=0)

        output_att = self.att(output, encoder_outputs)

        return output_att, (hidden, cell)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE * 2, bias=False)
        self.W2 = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE * 2, bias=False)
        self.v = nn.Linear(HIDDEN_SIZE * 2, 1, bias=False)

    def forward(self, key, value):
        # attention
        out1 = self.W1(key).unsqueeze(0)

        out2 = self.W2(value)
        # works thanks to broadcasting
        output = self.v(torch.tanh(out1 + out2)).squeeze(-1).T

        return output


class StateEmbedding(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE, padding_idx=padding_idx)
        self.rnn = nn.LSTM(input_size=EMBEDDING_SIZE,
                           hidden_size=HIDDEN_SIZE,
                           num_layers=1,
                           bidirectional=False)

    def representation(self, input, input_mask):
        embedded_input = self.embedding(input)
        output, (h, c) = self.rnn(embedded_input)

        # mean pooling, but take padding into account
        repr = output.sum(dim=0) / input_mask.sum(dim=0).unsqueeze(-1)
        return repr

    def forward(self, state, state_mask, cmds, cmds_mask):
        state_repr = self.representation(state, state_mask)  # n_batch, hidden
        cmds_repr = self.representation(cmds, cmds_mask)     # n_cmds, hidden

        n_batch = state_repr.size(0)
        n_cmds = cmds_repr.size(0)

        state_repr = state_repr.unsqueeze(0).repeat_interleave(n_cmds, 0)  # n_cmds, n_batch, hidden
        cmds_repr = cmds_repr.unsqueeze(1).repeat_interleave(n_batch, 1)  # n_cmds, n_batch, hidden

        output = torch.cat([state_repr, cmds_repr], dim=-1)  # n_cmds, n_batch, hidden * 2

        return output
