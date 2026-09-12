"""Download, clean, and split a parallel corpus."""

import argparse
import shutil
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from machine_translation import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the translation data.")
    parser.add_argument("--input", help="override the raw data path")
    parser.add_argument(
        "--config",
        help="experiment YAML (default: configs/seq2seq_tatoeba.yaml)",
    )
    return parser.parse_args()


def read_pairs(path, source_language, target_language):
    if path.suffix == ".parquet":
        raw = pd.read_parquet(path)
        if "translation" in raw.columns:
            return pd.DataFrame(raw["translation"].tolist())
        return raw

    if path.suffix == ".zip":
        with ZipFile(path) as archive:
            source_name = next(
                name
                for name in archive.namelist()
                if name.endswith(f".{source_language}")
            )
            target_name = next(
                name
                for name in archive.namelist()
                if name.endswith(f".{target_language}")
            )
            source_texts = archive.read(source_name).decode("utf-8").splitlines()
            target_texts = archive.read(target_name).decode("utf-8").splitlines()

        if len(source_texts) != len(target_texts):
            raise ValueError("The source and target files have different lengths")
        return pd.DataFrame(
            {
                source_language: source_texts,
                target_language: target_texts,
            }
        )

    raise ValueError("Input must be a Parquet file or an OPUS zip archive")


def main():
    args = parse_args()
    config = load_config(args.config)
    data_config = config.data
    raw_path = Path(args.input) if args.input else data_config.raw_path
    output_paths = [
        data_config.train_path,
        data_config.validation_path,
        data_config.test_path,
    ]

    if any(path.exists() for path in output_paths):
        answer = input("Replace the existing data splits? [y/N]: ").strip().lower()
        if answer not in {"y", "yes", "o", "oui"}:
            print("Data preparation cancelled.")
            return

    if not raw_path.exists():
        if data_config.download_url is None:
            raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

        print(f"Downloading {data_config.download_url}")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = raw_path.with_suffix(raw_path.suffix + ".part")
        with urlopen(data_config.download_url, timeout=60) as response:
            with temporary_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        temporary_path.replace(raw_path)

    source = data_config.source_language
    target = data_config.target_language
    pairs = read_pairs(
        raw_path,
        data_config.source_language,
        data_config.target_language,
    )
    pairs = pairs[[source, target]].dropna().copy()
    pairs[source] = pairs[source].str.replace(r"\s+", " ", regex=True).str.strip()
    pairs[target] = pairs[target].str.replace(r"\s+", " ", regex=True).str.strip()
    pairs = pairs[(pairs[source] != "") & (pairs[target] != "")]

    before = len(pairs)
    pairs = pairs.drop_duplicates().reset_index(drop=True)
    groups = pairs[source].str.casefold()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=data_config.split_seed,
    )
    train_indices, remaining_indices = next(splitter.split(pairs, groups=groups))
    train = pairs.iloc[train_indices]
    remaining = pairs.iloc[remaining_indices]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=data_config.split_seed,
    )
    validation_indices, test_indices = next(
        splitter.split(remaining, groups=groups.iloc[remaining_indices])
    )
    validation = remaining.iloc[validation_indices]
    test = remaining.iloc[test_indices]

    train_targets = set(train[target].str.casefold())
    validation = validation[
        ~validation[target].str.casefold().isin(train_targets)
    ]
    validation_targets = set(validation[target].str.casefold())
    test = test[
        ~test[target].str.casefold().isin(train_targets | validation_targets)
    ]

    if data_config.validation_size is not None:
        validation = validation.sample(
            n=data_config.validation_size,
            random_state=data_config.split_seed,
        )
    if data_config.test_size is not None:
        test = test.sample(
            n=data_config.test_size,
            random_state=data_config.split_seed,
        )

    for dataframe, path in zip((train, validation, test), output_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(path, index=False)

    print(f"Raw pairs: {before:,}")
    print(f"Removed duplicate pairs: {before - len(pairs):,}")
    print(f"Train: {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Test: {len(test):,}")
    print("Retrain the tokenizer and model after changing these splits.")


if __name__ == "__main__":
    main()
