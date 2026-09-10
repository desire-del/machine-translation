"""Train and save the shared tokenizer."""

import argparse

import pandas as pd

from machine_translation.config import load_data_config, load_tokenizer_config
from machine_translation.tokenization import build_tokenizer, save_tokenizer


DATA_CONFIG_PATH = "configs/data/default.yaml"
TOKENIZER_CONFIG_PATH = "configs/tokenizers/shared_unigram.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the shared tokenizer.")
    parser.add_argument(
        "--data-config",
        default=DATA_CONFIG_PATH,
        help="data config path (default: %(default)s)",
    )
    parser.add_argument(
        "--tokenizer-config",
        default=TOKENIZER_CONFIG_PATH,
        help="tokenizer config path (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config = load_data_config(args.data_config)
    tokenizer_config = load_tokenizer_config(args.tokenizer_config)
    columns = [tokenizer_config.source_column, tokenizer_config.target_column]

    pairs = pd.read_parquet(data_config.train_path, columns=columns)
    if tokenizer_config.deduplicate:
        pairs = pairs.drop_duplicates()

    texts = (
        text
        for pair in pairs.itertuples(index=False, name=None)
        for text in pair
    )
    tokenizer = build_tokenizer(texts, tokenizer_config)
    artifact_path = save_tokenizer(tokenizer, tokenizer_config.artifact_path)

    print(f"Training pairs: {len(pairs):,}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
    print(f"Tokenizer saved to: {artifact_path}")


if __name__ == "__main__":
    main()
