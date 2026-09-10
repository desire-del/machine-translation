# Educational Neural Machine Translation

A small project for learning neural machine translation by implementing three
models in sequence:

1. Seq2Seq with LSTM
2. Seq2Seq with Bahdanau attention
3. A small Transformer inspired by *Attention Is All You Need*

The final experiments compare the models using the same data and evaluation
setup. The model files are intentionally minimal so their components can be
implemented manually.

## Structure

```text
configs/
  data/                 Dataset configuration
  models/               One configuration per model
  tokenizers/           Shared tokenizer configuration
  training/             Shared training configuration
data/                 Local datasets (not committed)
notebooks/            Exploration and learning notebooks
src/machine_translation/
  config.py           Configuration loading and validation
  data.py             Data loading and preprocessing
  tokenization.py     Shared tokenizer construction and loading
  seq2seq.py          Seq2Seq LSTM model
  attention.py        Bahdanau attention model
  transformer.py      Small Transformer model
  evaluation.py       Shared evaluation helpers
scripts/              Tokenizer, training and comparison entry points
experiments/          Experiment notes and results
reports/              Final write-ups and figures
```

## Configuration

Configuration is split by responsibility:

- `configs/data/default.yaml` describes the dataset and preprocessing values.
- `configs/models/` contains one explicit file for each model architecture.
- `configs/tokenizers/shared_unigram.yaml` configures the shared tokenizer.
- `configs/training/default.yaml` contains the shared training values.

Paths in configuration files are relative to the repository root. Keep these
files under version control and do not store secrets or machine-specific paths
in them.

When the scripts are implemented, pass the configuration paths explicitly. For
example:

```bash
python scripts/train_seq2seq.py \
  --data-config configs/data/default.yaml \
  --model-config configs/models/seq2seq_lstm.yaml \
  --training-config configs/training/default.yaml
```

The loaders in `src/machine_translation/config.py` validate the data and
tokenizer files currently used by the project. Model and training loaders can
be added later when their scripts are implemented.

```python
from machine_translation.config import load_data_config, load_tokenizer_config

data_config = load_data_config("configs/data/default.yaml")
tokenizer_config = load_tokenizer_config(
    "configs/tokenizers/shared_unigram.yaml"
)
```

## Train the tokenizer

Train the shared tokenizer once from the configured training split:

```bash
uv run python scripts/train_tokenizer.py
```

This creates `artifacts/tokenizer/tokenizer.json`. The command refuses to
replace an existing tokenizer file.

The default configuration paths can be replaced from the command line:

```bash
uv run python scripts/train_tokenizer.py \
  --data-config configs/data/default.yaml \
  --tokenizer-config configs/tokenizers/shared_unigram.yaml
```

## Load the datasets

The main data API is available directly from the package:

```python
from machine_translation import (
    get_data_loaders,
    load_data_config,
    load_tokenizer,
    load_tokenizer_config,
)

data_config = load_data_config("configs/data/default.yaml")
tokenizer_config = load_tokenizer_config(
    "configs/tokenizers/shared_unigram.yaml"
)
tokenizer = load_tokenizer(
    tokenizer_config.artifact_path,
    tokenizer_config.special_tokens,
)

train_loader, validation_loader, test_loader = get_data_loaders(
    tokenizer,
    data_config,
    tokenizer_config,
)
batch = next(iter(train_loader))

print(batch["source_ids"].shape)
print(batch["source_padding_mask"].shape)
```

`get_data_loaders` returns the train, validation, and test loaders in that
order. Each loader returns a dictionary with the following contract:

| Attribute | Type and shape | Meaning |
| --- | --- | --- |
| `source_ids` | `torch.long[B, S]` | Padded source token IDs |
| `source_lengths` | `torch.long[B]` | Source lengths without padding |
| `source_padding_mask` | `torch.bool[B, S]` | `True` where the source is padding |
| `target_ids` | `torch.long[B, T]` | Padded target token IDs |
| `target_padding_mask` | `torch.bool[B, T]` | `True` where the target is padding |

`S` and `T` are padded independently to the longest sequence in the current
batch and never exceed `max_sequence_length`.

## Suggested order

Start with data preparation, implement and test each model in order, then use
the comparison script and notebooks to summarize the results.
