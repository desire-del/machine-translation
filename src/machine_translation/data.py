"""Parallel-text dataset and batch preparation."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from pandas import DataFrame
from tokenizers import Encoding, Tokenizer
from torch.utils.data import DataLoader, Dataset

from machine_translation.config import (
    DataConfig,
    SpecialTokensConfig,
    TokenizerConfig,
)
from machine_translation.tokenization import encode_batch, get_special_token_ids


class TranslationDataset(Dataset[tuple[str, str]]):
    """Store aligned source and target texts without tokenizing them."""

    def __init__(
        self,
        data_path: str | Path | DataFrame,
        source_column: str,
        target_column: str,
    ) -> None:
        if isinstance(data_path, (str, Path)):
            dataframe = pd.read_parquet(data_path)
        else:
            dataframe = data_path

        missing_columns = {source_column, target_column} - set(dataframe.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing translation column(s): {missing}")

        self.source_texts = dataframe[source_column].tolist()
        self.target_texts = dataframe[target_column].tolist()

    def __len__(self) -> int:
        return len(self.source_texts)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.source_texts[index], self.target_texts[index]


def _to_tensors(encodings: list[Encoding]) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([encoding.ids for encoding in encodings], dtype=torch.long)
    attention_mask = torch.tensor(
        [encoding.attention_mask for encoding in encodings],
        dtype=torch.bool,
    )
    return ids, attention_mask


def make_collate_fn(
    tokenizer: Tokenizer,
    special_tokens: SpecialTokensConfig,
    max_sequence_length: int,
) -> Callable[[list[tuple[str, str]]], dict[str, torch.Tensor]]:
    """Create the batch function.

    Returned shapes are ``source_ids[B, S]``, ``source_lengths[B]``,
    ``source_padding_mask[B, S]``, ``target_ids[B, T]`` and
    ``target_padding_mask[B, T]``. Padding masks are boolean and ``True`` marks
    positions to ignore.
    """
    if max_sequence_length < 2:
        raise ValueError("max_sequence_length must be at least 2")

    special_ids = get_special_token_ids(tokenizer, special_tokens)
    tokenizer.enable_truncation(max_length=max_sequence_length)
    tokenizer.enable_padding(
        direction="right",
        pad_id=special_ids["pad"],
        pad_token=special_tokens.pad,
    )

    def collate_fn(batch: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        source_texts, target_texts = zip(*batch)
        source_encodings = encode_batch(
            tokenizer,
            source_texts,
        )
        target_encodings = encode_batch(
            tokenizer,
            target_texts,
        )

        source_ids, source_attention_mask = _to_tensors(source_encodings)
        target_ids, target_attention_mask = _to_tensors(target_encodings)

        return {
            "source_ids": source_ids,
            "source_lengths": source_attention_mask.sum(dim=1),
            "source_padding_mask": ~source_attention_mask,
            "target_ids": target_ids,
            "target_padding_mask": ~target_attention_mask,
        }

    return collate_fn


def get_data_loaders(
    tokenizer: Tokenizer,
    data_config: DataConfig,
    tokenizer_config: TokenizerConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return ``(train_loader, validation_loader, test_loader)``."""
    collate_fn = make_collate_fn(
        tokenizer,
        tokenizer_config.special_tokens,
        data_config.max_sequence_length,
    )

    def create_loader(data_path: Path, shuffle: bool) -> DataLoader:
        dataset = TranslationDataset(
            data_path,
            tokenizer_config.source_column,
            tokenizer_config.target_column,
        )
        return DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            shuffle=shuffle,
            num_workers=data_config.num_workers,
            collate_fn=collate_fn,
        )

    return (
        create_loader(data_config.train_path, shuffle=True),
        create_loader(data_config.validation_path, shuffle=False),
        create_loader(data_config.test_path, shuffle=False),
    )
