# Neural Machine Translation

An educational English-to-French machine translation project built from
first principles with PyTorch. The goal is to study how conditional sequence
generation evolves from recurrent Seq2Seq models to attention-based models and
Transformers through controlled, reproducible experiments.

The repository intentionally favors explicit model code and a small project
structure over training frameworks and production infrastructure.

## Project status

| Stage | Status | Main question |
| --- | --- | --- |
| Seq2Seq LSTM | Implemented | How limiting is a fixed-size context vector? |
| Bahdanau attention | Planned | Does attention improve long-sequence translation? |
| Small Transformer | Planned | Can self-attention replace recurrence? |
| Comparative benchmark | Planned | What are the quality and compute trade-offs? |

The current baseline includes:

- separate English and French BPE tokenizers;
- dynamic batch padding and packed source sequences;
- configurable teacher forcing and gradient clipping;
- validation loss, perplexity, early stopping, and resumable checkpoints;
- greedy decoding and length-normalized beam search;
- reproducible runs organized by experiment name and random seed.

## Experimental pipeline

```text
OPUS Tatoeba corpus
        ↓
clean train / validation / test splits
        ↓
English BPE + French BPE tokenizers
        ↓
PyTorch DataLoaders
        ↓
Seq2Seq → Attention → Transformer
        ↓
comparable experiment artifacts and analyses
```

The default experiment uses the English-French portion of the OPUS Tatoeba
corpus. Data preparation removes empty and duplicate pairs and prevents the
same source or target sentence from leaking across splits.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) for dependency and virtual
environment management.

```bash
uv sync
```

Install the notebook dependencies when needed:

```bash
uv sync --group dev
```

Check whether the installed PyTorch build can access the GPU:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Training automatically selects CUDA when available. It can also be selected
explicitly with `--device cuda`.

## Quick start

Run the complete baseline workflow from the repository root:

```bash
uv run python scripts/prepare_data.py
uv run python scripts/train_tokenizer.py
uv run python scripts/train_seq2seq.py
uv run python scripts/translate.py "I love cats."
```

The data and tokenizer scripts ask for confirmation before replacing existing
artifacts.

### Prepare the dataset

```bash
uv run python scripts/prepare_data.py
```

This command downloads the configured corpus when it is missing, cleans the
parallel text, and writes reproducible Parquet splits under `data/processed/`.
Use `--input` to prepare a local Parquet file or compatible OPUS archive.

### Train the tokenizers

```bash
uv run python scripts/train_tokenizer.py
```

The default configuration trains one 8,000-token BPE vocabulary for English
and another for French. Each saved `tokenizer.json` contains its normalization,
pre-tokenization, BOS/EOS processing, vocabulary, and decoding behavior. Dynamic
padding is applied later by the DataLoader.

### Train Seq2Seq

```bash
uv run python scripts/train_seq2seq.py
```

Training saves the best validation checkpoint and the most recent checkpoint.
To run a temporary trial without writing an experiment directory:

```bash
uv run python scripts/train_seq2seq.py --no-save
```

Resume the most recent checkpoint for a saved run:

```bash
uv run python scripts/train_seq2seq.py --resume
```

### Translate

```bash
uv run python scripts/translate.py "The student reads a book."
```

Beam search is used by default. Greedy decoding is available with:

```bash
uv run python scripts/translate.py "The student reads a book." --beam-size 1
```

## Configuration

One self-contained YAML file describes one experimental condition. The default
configuration is [`configs/seq2seq_tatoeba.yaml`](configs/seq2seq_tatoeba.yaml).

```yaml
name: seq2seq_tatoeba
seed: 42

data:
  split_seed: 42
  max_sequence_length: 64
  batch_size: 64

tokenizer:
  type: bpe
  shared: false
  vocab_size: 8000

model:
  type: seq2seq
  embedding_dim: 256
  hidden_dim: 512
  num_layers: 2
  dropout: 0.2

training:
  epochs: 10
  learning_rate: 0.001
  teacher_forcing_ratio: 1.0
  gradient_clip_norm: 1.0
  patience: 3
  min_delta: 0.0
```

All paths are resolved relative to the repository root. Configuration values
are validated before any data preparation or training begins.

`data.split_seed` controls the dataset split and should remain fixed when
models are compared. The top-level `seed` controls model initialization and
batch ordering. This separation allows repeated training runs on identical
data.

Early stopping counts validation checks without an improvement greater than
`min_delta`. Set `patience: null` to disable it.

Use another configuration with:

```bash
uv run python scripts/train_seq2seq.py --config configs/my_experiment.yaml
```

## Experiment management

Override only the run seed to repeat the same experimental condition:

```bash
uv run python scripts/train_seq2seq.py --seed 42
uv run python scripts/train_seq2seq.py --seed 123
uv run python scripts/train_seq2seq.py --seed 456
```

Saved runs follow this structure:

```text
experiments/seq2seq_tatoeba/seed_42/
├── config.yaml
├── history.json
├── metadata.json
├── metrics.json
└── checkpoints/
    ├── best.pt
    └── last.pt
```

| Artifact | Contents |
| --- | --- |
| `config.yaml` | Effective configuration used by the run |
| `history.json` | Training and validation curves by epoch |
| `metadata.json` | PyTorch, CUDA, device, GPU, and parameter information |
| `metrics.json` | Best validation epoch, loss, perplexity, and training time |
| `best.pt` | Model with the best validation loss |
| `last.pt` | Latest model and optimizer state for resuming |

Existing runs are never overwritten silently. Select another seed, resume the
run, or use `--no-save` for a disposable trial.

## Evaluation protocol

Development currently uses token-level validation loss and perplexity for
model selection and early stopping. The final comparison will evaluate all
models on the same untouched test split and report:

- BLEU and chrF;
- quality by source-sentence length;
- parameter count and training time;
- greedy and beam-search inference latency;
- qualitative errors such as omissions, repetitions, and mistranslated names.

No test-set benchmark is reported yet because the repository is still at the
Seq2Seq baseline stage.

## Data contract

Every DataLoader batch is a dictionary with the same public structure:

| Key | Shape | Description |
| --- | --- | --- |
| `source_ids` | `[B, S]` | Padded source token IDs |
| `source_lengths` | `[B]` | Source lengths before padding |
| `source_padding_mask` | `[B, S]` | `True` at source padding positions |
| `target_ids` | `[B, T]` | Padded target token IDs |
| `target_padding_mask` | `[B, T]` | `True` at target padding positions |

`S` and `T` are determined independently by the longest sequence in each
batch and never exceed `max_sequence_length`.

## Repository structure

```text
.
├── configs/                     Experiment definitions
├── data/                        Raw and processed corpora (not tracked)
├── artifacts/                   Trained tokenizers (not tracked)
├── experiments/                 Run artifacts and checkpoints
├── notebooks/
│   ├── 01_data.ipynb            Data and batch inspection
│   └── 02_seq2seq_results.ipynb Seq2Seq results and translations
├── scripts/
│   ├── prepare_data.py
│   ├── train_tokenizer.py
│   ├── train_seq2seq.py
│   └── translate.py
└── src/machine_translation/
    ├── config.py
    ├── data.py
    ├── tokenization.py
    ├── seq2seq.py
    ├── training.py
    └── inference.py
```

Model implementations live in `src/`. Notebooks are reserved for inspection,
visualization, and experimental analysis rather than core model logic.

## Roadmap

- [x] Reproducible English-French dataset preparation
- [x] Separate BPE tokenizers
- [x] Vanilla Seq2Seq LSTM baseline
- [x] Greedy and beam-search decoding
- [x] Saved runs, checkpoint resume, and early stopping
- [ ] Bahdanau attention and alignment visualizations
- [ ] Small encoder-decoder Transformer
- [ ] BLEU and chrF evaluation on the held-out test set
- [ ] Quality by sentence length and decoding latency
- [ ] Cross-model error analysis and final benchmark

## References

- Sutskever, Vinyals, and Le, [Sequence to Sequence Learning with Neural
  Networks](https://arxiv.org/abs/1409.3215), 2014.
- Bahdanau, Cho, and Bengio, [Neural Machine Translation by Jointly Learning to
  Align and Translate](https://arxiv.org/abs/1409.0473), 2014.
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
  2017.
