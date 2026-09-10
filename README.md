# Educational Neural Machine Translation

A small project for learning neural machine translation by implementing three
models in sequence:

1. Seq2Seq with LSTM
2. Seq2Seq with Bahdanau attention
3. A small Transformer inspired by *Attention Is All You Need*

The final experiments compare the models using the same data and evaluation
setup. The source files are intentionally empty so the model components can be
implemented manually.

## Structure

```text
configs/
  data/                 Dataset configuration
  models/               One configuration per model
  training/             Shared training configuration
data/                 Local datasets (not committed)
notebooks/            Exploration and learning notebooks
src/machine_translation/
  config.py           Configuration loading and validation
  data.py             Data loading and preprocessing
  seq2seq.py          Seq2Seq LSTM model
  attention.py        Bahdanau attention model
  transformer.py      Small Transformer model
  evaluation.py       Shared evaluation helpers
scripts/              Training and comparison entry points
experiments/          Experiment notes and results
reports/              Final write-ups and figures
```

## Configuration

Configuration is split by responsibility:

- `configs/data/default.yaml` describes the dataset and preprocessing values.
- `configs/models/` contains one explicit file for each model architecture.
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

The loader in `src/machine_translation/config.py` parses and validates every
file before a dataset or model is created. Unknown keys and invalid values are
rejected, and paths declared in YAML are resolved from the repository root.

```python
from machine_translation.config import load_config

config = load_config(
    "configs/data/default.yaml",
    "configs/models/seq2seq_lstm.yaml",
    "configs/training/default.yaml",
)
```

Each experiment should also save `config.model_dump(mode="json")` beside its
results so that the run can be reproduced later.

## Suggested order

Start with data preparation, implement and test each model in order, then use
the comparison script and notebooks to summarize the results.
