"""Seq2Seq LSTM model with separate source and target vocabularies."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from machine_translation.config import ModelConfig
from einops import rearrange

# B: batch size
# S: source sequence length
# U: target sequence length
# E: embedding dimension
# H: hidden dimension
# A: attention dimension
# L: number of LSTM layers
# V: target vocabulary size


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pad_id,
        hidden_dim,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional= True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.hidden_proj = nn.Linear(hidden_dim*2, hidden_dim)
        self.cell_proj = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, source_ids, source_lengths):
        # source_ids: [B, S]
        # source_lengths: [B]
        embedded = self.dropout(self.embedding(source_ids)) # [B, S, E]
        packed = pack_padded_sequence(
                    embedded,
                    source_lengths.cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
        packed_states, (hidden, cell) = self.lstm(packed)
        # packed_states: PackedSequence containing states of dimension 2H
        # hidden, cell: [2L, B, H]
        encoder_states, _ = pad_packed_sequence(
          packed_states,
          batch_first=True,
          total_length=source_ids.size(1),
        ) # encoder_states: [B, S, 2H]
        hidden = rearrange(hidden, "(l d) b h -> l b (d h)", l=self.num_layers, d=2) # [L, B, 2H]
        cell = rearrange(cell, "(l d) b h -> l b (d h)", l=self.num_layers, d=2) # [L, B, 2H]
        hidden = self.hidden_proj(hidden) # [L, B, H]
        cell = self.cell_proj(cell) # [L, B, H]

        return encoder_states, hidden, cell

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, att_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.att_dim = att_dim

        self.encoder_hiddens_proj = nn.Linear(hidden_dim*2 , att_dim)
        self.decoder_hidden_proj = nn.Linear(hidden_dim, att_dim)
        self.att_proj = nn.Linear(att_dim, 1)

    def project_encoder(self, encode_states):
        """Project source states once for all decoder steps."""
        return self.encoder_hiddens_proj(encode_states)

    def forward(self, encode_states, decoder_hidden, source_padding_mask, x1=None):
        # encode_states: [B, S, 2H]
        # decoder_hidden: [B, 1, H]
        # source_padding_mask: [B, S], True at padding positions
        source_padding_mask = source_padding_mask.to(
            device=encode_states.device,
            dtype=torch.bool,
        )
        if x1 is None:
            x1 = self.project_encoder(encode_states)
        # x1: [B, S, A]
        x2 = self.decoder_hidden_proj(decoder_hidden) # [B, 1, A]

        x = torch.tanh(x1 + x2) # [B, S, A], using broadcasting
        scores = self.att_proj(x).squeeze(-1) # [B, S]
        scores = scores.masked_fill(source_padding_mask, float("-inf")) # [B, S]

        alpha = torch.softmax(scores, dim=-1) # [B, S]

        context = torch.bmm(alpha.unsqueeze(1), encode_states)
        # [B, 1, S] @ [B, S, 2H] -> [B, 1, 2H]

        return alpha, context




class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout,
        att_dim
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id,
        )
        self.dropout = nn.Dropout(dropout)
        self.att = BahdanauAttention(
            hidden_dim=hidden_dim,
            att_dim=att_dim
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim*2 + embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        encoder_states,
        source_padding_mask,
        target_ids,
        hidden,
        cell,
        x1=None,
    ):
        # target_ids: [B, 1]
        # hidden, cell: current decoder states [L, B, H]
        embedded  = self.dropout(self.embedding(target_ids)) # [B, 1, E]
        query = hidden[-1].unsqueeze(1) # [B, 1, H]
        alpha, context = self.att(
            encoder_states,
            query,
            source_padding_mask,
            x1=x1,
        )
        # alpha: [B, S]; context: [B, 1, 2H]
        lstm_input = torch.cat([embedded, context], dim=-1) # [B, 1, E + 2H]
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # outputs: [B, 1, H]; hidden, cell: [L, B, H]
        # self.output_layer(outputs): [B, 1, V]
        return self.output_layer(outputs), hidden, cell, alpha


class Seq2SeqAtt(nn.Module):
    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        source_pad_id,
        target_pad_id,
        config: ModelConfig,
    ):
        super().__init__()

        if config.type != "seq2seq_att":
            raise ValueError(
                "Seq2SeqAtt requires a model config with type 'seq2seq_att'"
            )
        att_dim = config.att_dim

        self.encoder = Encoder(
            vocab_size=source_vocab_size,
            embedding_dim=config.embedding_dim,
            pad_id=source_pad_id,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.decoder = Decoder(
            vocab_size=target_vocab_size,
            pad_id=target_pad_id,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            att_dim=att_dim
        )

    def forward(
        self,
        source_ids,
        source_lengths,
        source_padding_mask,
        target_ids,
        teacher_forcing_ratio=0.5,
    ):
        if not 0 <= teacher_forcing_ratio <= 1:
            raise ValueError("teacher_forcing_ratio must be between 0 and 1")
        if target_ids.size(1) < 2:
            raise ValueError("target_ids must contain at least <bos> and <eos>")

        encoder_states, hidden, cell = self.encoder(source_ids, source_lengths)
        x1 = self.decoder.att.project_encoder(encoder_states) # [B, S, A]

        decoder_input = target_ids[:, 0].unsqueeze(1) # [B, 1]
        logits = []
        alphas = []

        for step in range(1, target_ids.size(1)):
            step_logits, hidden, cell, alpha = self.decoder(
                encoder_states,
                source_padding_mask,
                decoder_input,
                hidden,
                cell,
                x1=x1,
            )
            step_logits = step_logits.squeeze(1) # [B, V]
            # alpha: [B, S]
            alphas.append(alpha)
            logits.append(step_logits)

            predictions = step_logits.argmax(dim=1)
            use_target = torch.rand(
                target_ids.size(0),
                device=target_ids.device,
            ) < teacher_forcing_ratio
            next_tokens = torch.where(
                use_target,
                target_ids[:, step],
                predictions,
            )
            decoder_input = next_tokens.unsqueeze(1)

        # torch.stack(logits): [B, U - 1, V]
        # torch.stack(alphas): [B, U - 1, S]
        return torch.stack(logits, dim=1), torch.stack(alphas, dim=1)


def build_seq2seqatt(
    source_tokenizer,
    target_tokenizer,
    special_tokens,
    model_config,
):
    """Build Seq2SeqAtt from tokenizers and its validated model config."""
    return Seq2SeqAtt(
        source_vocab_size=source_tokenizer.get_vocab_size(),
        target_vocab_size=target_tokenizer.get_vocab_size(),
        source_pad_id=source_tokenizer.token_to_id(special_tokens.pad),
        target_pad_id=target_tokenizer.token_to_id(special_tokens.pad),
        config=model_config,
    )
