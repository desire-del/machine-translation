"""Build the configured data loaders and print a short summary."""

from machine_translation import (
    get_data_loaders,
    load_data_config,
    load_tokenizer,
    load_tokenizer_config,
)


DATA_CONFIG_PATH = "configs/data/default.yaml"
TOKENIZER_CONFIG_PATH = "configs/tokenizers/shared_unigram.yaml"


def main() -> None:
    data_config = load_data_config(DATA_CONFIG_PATH)
    tokenizer_config = load_tokenizer_config(TOKENIZER_CONFIG_PATH)
    tokenizer = load_tokenizer(
        tokenizer_config.artifact_path,
        tokenizer_config.special_tokens,
    )
    train_loader, validation_loader, test_loader = get_data_loaders(
        tokenizer,
        data_config,
        tokenizer_config,
    )

    print(f"Train batches: {len(train_loader):,}")
    print(f"Validation batches: {len(validation_loader):,}")
    print(f"Test batches: {len(test_loader):,}")


if __name__ == "__main__":
    main()
