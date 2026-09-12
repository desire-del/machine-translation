"""Seq2Seq LSTM model with separate source and target vocabularies."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from machine_translation.config import ModelConfig


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id,
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, source_ids, source_lengths):
        embedded = self.dropout(self.embedding(source_ids))
        packed = pack_padded_sequence(
            embedded,
            source_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, cell) = self.lstm(packed)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id,
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, target_ids, hidden, cell):
        embedded = self.dropout(self.embedding(target_ids))
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        return self.output_layer(outputs), hidden, cell


class Seq2Seq(nn.Module):
    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        source_pad_id,
        target_pad_id,
        config: ModelConfig,
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=source_vocab_size,
            pad_id=source_pad_id,
            embedding_dim=config.embedding_dim,
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
        )

    def forward(
        self,
        source_ids,
        source_lengths,
        target_ids,
        teacher_forcing_ratio=0.5,
    ):
        if not 0 <= teacher_forcing_ratio <= 1:
            raise ValueError("teacher_forcing_ratio must be between 0 and 1")
        if target_ids.size(1) < 2:
            raise ValueError("target_ids must contain at least <bos> and <eos>")

        hidden, cell = self.encoder(source_ids, source_lengths)

        # With full teacher forcing, all target tokens can be decoded at once.
        if teacher_forcing_ratio == 1:
            logits, _, _ = self.decoder(target_ids[:, :-1], hidden, cell)
            return logits

        decoder_input = target_ids[:, 0].unsqueeze(1)
        logits = []

        for step in range(1, target_ids.size(1)):
            step_logits, hidden, cell = self.decoder(
                decoder_input,
                hidden,
                cell,
            )
            step_logits = step_logits.squeeze(1)
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

        return torch.stack(logits, dim=1)


def build_seq2seq(
    source_tokenizer,
    target_tokenizer,
    special_tokens,
    model_config,
):
    """Build Seq2Seq with vocabulary sizes and pad IDs from the tokenizers."""
    return Seq2Seq(
        source_vocab_size=source_tokenizer.get_vocab_size(),
        target_vocab_size=target_tokenizer.get_vocab_size(),
        source_pad_id=source_tokenizer.token_to_id(special_tokens.pad),
        target_pad_id=target_tokenizer.token_to_id(special_tokens.pad),
        config=model_config,
    )
