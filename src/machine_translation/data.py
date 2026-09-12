"""Parallel-text dataset and data loaders."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from machine_translation.tokenization import encode_batch, get_special_token_ids


class TranslationDataset(Dataset):
    """Store aligned source and target texts without tokenizing them."""

    def __init__(self, data, source_column, target_column):
        dataframe = pd.read_parquet(data) if isinstance(data, (str, Path)) else data

        missing_columns = {source_column, target_column} - set(dataframe.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing translation column(s): {missing}")

        self.source_texts = dataframe[source_column].tolist()
        self.target_texts = dataframe[target_column].tolist()

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, index):
        return self.source_texts[index], self.target_texts[index]


def _to_tensors(encodings):
    ids = torch.tensor([encoding.ids for encoding in encodings], dtype=torch.long)
    mask = torch.tensor(
        [encoding.attention_mask for encoding in encodings],
        dtype=torch.bool,
    )
    return ids, mask


def make_collate_fn(
    source_tokenizer,
    target_tokenizer,
    special_tokens,
    max_sequence_length,
):
    """Create batches with padded IDs, lengths, and padding masks.

    Shapes: source IDs ``[B, S]``, source lengths ``[B]``, source padding
    mask ``[B, S]``, target IDs ``[B, T]`` and target padding mask ``[B, T]``.
    A padding mask is boolean and ``True`` means "ignore this position".
    """
    if max_sequence_length < 2:
        raise ValueError("max_sequence_length must be at least 2")

    for tokenizer in (source_tokenizer, target_tokenizer):
        special_ids = get_special_token_ids(tokenizer, special_tokens)
        tokenizer.enable_truncation(max_length=max_sequence_length)
        tokenizer.enable_padding(
            direction="right",
            pad_id=special_ids["pad"],
            pad_token=special_tokens.pad,
        )

    def collate(batch):
        source_texts, target_texts = zip(*batch)
        source_ids, source_mask = _to_tensors(
            encode_batch(source_tokenizer, source_texts)
        )
        target_ids, target_mask = _to_tensors(
            encode_batch(target_tokenizer, target_texts)
        )

        return {
            "source_ids": source_ids,
            "source_lengths": source_mask.sum(dim=1),
            "source_padding_mask": ~source_mask,
            "target_ids": target_ids,
            "target_padding_mask": ~target_mask,
        }

    return collate


def get_data_loaders(
    source_tokenizer,
    target_tokenizer,
    data_config,
    special_tokens,
):
    """Return ``(train_loader, validation_loader, test_loader)``."""
    collate = make_collate_fn(
        source_tokenizer,
        target_tokenizer,
        special_tokens,
        data_config.max_sequence_length,
    )

    def create_loader(data_path, shuffle):
        dataset = TranslationDataset(
            data_path,
            data_config.source_language,
            data_config.target_language,
        )
        return DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            shuffle=shuffle,
            num_workers=data_config.num_workers,
            collate_fn=collate,
        )

    return (
        create_loader(data_config.train_path, shuffle=True),
        create_loader(data_config.validation_path, shuffle=False),
        create_loader(data_config.test_path, shuffle=False),
    )
