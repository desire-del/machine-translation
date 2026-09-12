"""Load one self-contained experiment configuration."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = PROJECT_ROOT / "configs/seq2seq_tatoeba.yaml"


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(ConfigModel):
    dataset: str
    split_seed: int = Field(ge=0)
    source_language: str
    target_language: str
    raw_path: Path
    download_url: str | None = None
    train_path: Path
    validation_path: Path
    test_path: Path
    validation_size: int | None = Field(default=None, gt=0)
    test_size: int | None = Field(default=None, gt=0)
    max_sequence_length: int = Field(gt=1)
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_languages(self):
        if self.source_language == self.target_language:
            raise ValueError("source and target languages must be different")
        return self


class SpecialTokensConfig(ConfigModel):
    pad: str
    unk: str
    bos: str
    eos: str

    @model_validator(mode="after")
    def validate_unique_tokens(self):
        if len(set(self.as_tuple())) != 4:
            raise ValueError("special tokens must be distinct")
        return self

    def as_tuple(self):
        return self.pad, self.unk, self.bos, self.eos


class TokenizerConfig(ConfigModel):
    type: Literal["unigram", "bpe"]
    shared: bool
    vocab_size: int = Field(gt=4)
    min_frequency: int = Field(default=2, gt=0)
    deduplicate: bool = True
    source_artifact_path: Path
    target_artifact_path: Path
    special_tokens: SpecialTokensConfig

    @model_validator(mode="after")
    def validate_artifact_paths(self):
        same_path = self.source_artifact_path == self.target_artifact_path
        if self.shared and not same_path:
            raise ValueError("shared tokenizers must use the same artifact path")
        if not self.shared and same_path:
            raise ValueError("separate tokenizers must use different artifact paths")
        return self


class ModelConfig(ConfigModel):
    type: Literal["seq2seq"]
    embedding_dim: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)


class TrainingConfig(ConfigModel):
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    teacher_forcing_ratio: float = Field(ge=0, le=1)
    gradient_clip_norm: float = Field(gt=0)
    evaluate_every: int = Field(gt=0)
    patience: int | None = Field(default=None, gt=0)
    min_delta: float = Field(default=0.0, ge=0)


class Config(ConfigModel):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    seed: int = Field(ge=0)
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    training: TrainingConfig

    @property
    def run_directory(self):
        return PROJECT_ROOT / "experiments" / self.name / f"seed_{self.seed}"

    @property
    def checkpoint_directory(self):
        return self.run_directory / "checkpoints"

    def snapshot(self):
        """Return a portable dictionary for run artifacts and checkpoints."""
        content = self.model_dump(mode="json")
        paths = {
            "data": ("raw_path", "train_path", "validation_path", "test_path"),
            "tokenizer": ("source_artifact_path", "target_artifact_path"),
        }

        for section, names in paths.items():
            for name in names:
                path = Path(content[section][name])
                try:
                    content[section][name] = str(path.relative_to(PROJECT_ROOT))
                except ValueError:
                    pass

        return content


def _resolve(path):
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path=None, seed=None):
    """Load and validate one complete experiment YAML file."""
    config_path = _resolve(Path(path)) if path else _DEFAULT_CONFIG

    try:
        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = Config.model_validate(content)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ValueError(f"Invalid configuration {config_path}:\n{error}") from error

    data = config.data.model_copy(
        update={
            name: _resolve(getattr(config.data, name))
            for name in ("raw_path", "train_path", "validation_path", "test_path")
        }
    )
    tokenizer = config.tokenizer.model_copy(
        update={
            "source_artifact_path": _resolve(
                config.tokenizer.source_artifact_path
            ),
            "target_artifact_path": _resolve(
                config.tokenizer.target_artifact_path
            ),
        }
    )

    if seed is not None and seed < 0:
        raise ValueError("seed must be greater than or equal to zero")

    config = config.model_copy(
        update={
            "data": data,
            "tokenizer": tokenizer,
        }
    )
    return config if seed is None else config.model_copy(update={"seed": seed})
