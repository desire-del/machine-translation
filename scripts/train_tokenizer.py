"""Train and save the configured tokenizer."""

import argparse

import pandas as pd

from machine_translation import load_config
from machine_translation.tokenization import build_tokenizer, save_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train the tokenizer.")
    parser.add_argument(
        "--config",
        help="experiment YAML (default: configs/seq2seq_tatoeba.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    data = config.data
    tokens = config.tokenizer
    source = data.source_language
    target = data.target_language
    columns = [source, target]
    pairs = pd.read_parquet(data.train_path, columns=columns)
    paths = list(
        dict.fromkeys([tokens.source_artifact_path, tokens.target_artifact_path])
    )
    overwrite = False

    existing = [path for path in paths if path.exists()]
    if existing:
        print("Existing tokenizer artifacts:")
        for path in existing:
            print(f"- {path}")

        answer = input("Replace them? [y/N]: ").strip().lower()
        if answer not in {"y", "yes", "o", "oui"}:
            print("Tokenizer training cancelled.")
            return
        overwrite = True

    def train(texts):
        if tokens.deduplicate:
            texts = texts.drop_duplicates()
        return build_tokenizer(
            texts=texts,
            vocab_size=tokens.vocab_size,
            special_tokens=tokens.special_tokens,
            model_type=tokens.type,
            min_frequency=tokens.min_frequency,
        )

    if tokens.shared:
        texts = pd.concat(
            [pairs[source], pairs[target]],
            ignore_index=True,
        )
        tokenizer = train(texts)
        path = save_tokenizer(tokenizer, paths[0], overwrite=overwrite)
        print(f"Shared vocabulary: {tokenizer.get_vocab_size():,} -> {path}")
    else:
        source_tokenizer = train(pairs[source])
        target_tokenizer = train(pairs[target])
        source_path = save_tokenizer(
            source_tokenizer,
            tokens.source_artifact_path,
            overwrite=overwrite,
        )
        target_path = save_tokenizer(
            target_tokenizer,
            tokens.target_artifact_path,
            overwrite=overwrite,
        )
        print(
            f"Source vocabulary: {source_tokenizer.get_vocab_size():,}"
            f" -> {source_path}"
        )
        print(
            f"Target vocabulary: {target_tokenizer.get_vocab_size():,}"
            f" -> {target_path}"
        )

    print(f"Training pairs: {len(pairs):,}")


if __name__ == "__main__":
    main()
