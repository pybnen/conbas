import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
#  from data import SOS_TOKEN_ID, EOS_TOKEN_ID


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, device, padding_idx):
        super().__init__()

        self.state_embedding = StateEmbedding(vocab_size, embedding_size, hidden_size, padding_idx)
        state_embedding_size = hidden_size * 2
        self.encoder = Encoder(state_embedding_size)
        self.seq_decoder = SeqDecoder(state_embedding_size, device)

    def forward(self, state, state_mask, cmds, cmds_mask):
        # n_batch, n_cmds, hidden * 2
        state_embed, lengths = self.state_embedding(state, state_mask, cmds, cmds_mask)

        # n_batch, n_cmds, hidden * 2
        encoder_outputs, hidden_states = self.encoder(state_embed, lengths)

        outputs, idxs = self.seq_decoder(encoder_outputs, hidden_states, lengths)

        return outputs, idxs


class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.rnn = nn.LSTM(input_size=hidden_size,
                           hidden_size=hidden_size,
                           num_layers=1,
                           bidirectional=False)

    def forward(self, input: torch.Tensor, lengths: torch.Tensor):
        # n_seq, n_batch, hidden
        input = input.permute((1, 0, 2))
        packed_input = pack_padded_sequence(input, lengths, enforce_sorted=False)

        output, (hidden, cell) = self.rnn(packed_input)
        output_padded, _ = pad_packed_sequence(output)

        # n_batch, n_seq, hidden
        output_padded = output_padded.permute((1, 0, 2))
        return output_padded, (hidden, cell)


class SeqDecoder(nn.Module):
    def __init__(self, hidden_size, device):
        super().__init__()

        self.decoder = Decoder(hidden_size)

        self.device = device

    def forward(self, encoder_outputs, hidden_state, lengths):
        n_batch, seq_len, n_hidden = encoder_outputs.size()

        # SOS
        decoder_input = torch.zeros_like(encoder_outputs[:, 0])

        done = torch.zeros(n_batch).bool().to(self.device)
        lengths = lengths.to(self.device)

        outputs = []
        indices = []
        batch_idx = torch.arange(n_batch)

        # TODO make smarter
        mask = torch.zeros(n_batch, seq_len).to(self.device)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        mask = mask.bool()
        already_sampled = torch.zeros(n_batch, seq_len).to(self.device).bool()

        cur_len = 0
        while not torch.all(done):
            # n_batch, n_cmds
            output_att, hidden_state = self.decoder(decoder_input, hidden_state, encoder_outputs)
            output_att[~mask] = float("-inf")

            masked_output_att = torch.clone(output_att.detach())
            masked_output_att[already_sampled] = float("-inf")

            dist = Categorical(logits=masked_output_att)
            input_idx = dist.sample()
            input_idx = input_idx * (1 - done.long())

            already_sampled[batch_idx, input_idx] = True
            already_sampled[batch_idx, 0] = False

            decoder_input = encoder_outputs[batch_idx, input_idx]
            # would also work:
            # index_tensor = input_idx.unsqueeze(1).unsqueeze(-1).expand(n_batch, 1, n_hidden)
            # decoder_input = encoder_outputs.gather(dim=1, index=index_tensor).squeeze(1)
            decoder_input = decoder_input.detach()

            indices.append(input_idx)
            outputs.append(output_att)

            cur_len += 1
            max_seq_len = cur_len >= lengths
            # EOF == idx 0
            done = torch.logical_or(done, torch.logical_or(max_seq_len, input_idx == 0))

        outputs = torch.stack(outputs, dim=1)
        indices = torch.stack(indices, dim=1)
        return outputs, indices


class Decoder(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.rnn = nn.LSTM(input_size=hidden_size,
                           hidden_size=hidden_size,
                           num_layers=1,
                           bidirectional=False)
        self.att = Attention(hidden_size)

    def forward(self, input: torch.Tensor, hidden_state: torch.Tensor, encoder_outputs: torch.Tensor):
        hidden, cell = hidden_state

        output, (hidden, cell) = self.rnn(input.unsqueeze(dim=0), (hidden, cell))
        output = output.squeeze(dim=0)

        output_att = self.att(output, encoder_outputs)

        return output_att, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, key, value):
        # attention
        out1 = self.W1(key).unsqueeze(1)

        out2 = self.W2(value)
        # works thanks to broadcasting
        output = self.v(torch.tanh(out1 + out2)).squeeze(-1)

        return output


class StateEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=hidden_size,
                           num_layers=1,
                           bidirectional=False)

    def state_representation(self, input, input_mask):
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

    def forward(self, state, state_mask, cmds, cmds_mask):
        # n_batch, hidden
        state_repr = self.state_representation(state, state_mask)

        outputs = []
        output_lengths = []
        for cmd, cmd_mak, s_rep in zip(cmds, cmds_mask, state_repr):
            # n_cmds[i], hidden
            cmd_rep = self.state_representation(cmd, cmd_mak)

            n_cmds, hidden = cmd_rep.size()
            output_lengths.append(n_cmds)

            # n_cmds[i], hidden
            s_rep_expand = s_rep.unsqueeze(0).expand(n_cmds, hidden)
            outputs.append(torch.cat([s_rep_expand, cmd_rep], dim=-1))

        # n_batch, n_cmds, hidden * 2
        outputs_padded = pad_sequence(outputs, batch_first=True)
        output_lengths = torch.tensor(output_lengths)
        return outputs_padded, output_lengths
