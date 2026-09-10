"""Load and validate the project's YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


PositiveInteger = Annotated[int, Field(gt=0)]
NonNegativeInteger = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
DropoutRate = Annotated[float, Field(ge=0, lt=1)]
LanguageCode = Annotated[str, Field(min_length=2)]


class ConfigError(ValueError):
    """Raised when a configuration file cannot be loaded or validated."""


class StrictConfigModel(BaseModel):
    """Base class shared by all immutable configuration models."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class DataConfig(StrictConfigModel):
    """Dataset and preprocessing configuration."""

    name: str = Field(min_length=1)
    source_language: LanguageCode
    target_language: LanguageCode
    train_path: Path
    validation_path: Path
    test_path: Path
    max_sequence_length: PositiveInteger
    min_token_frequency: PositiveInteger
    batch_size: PositiveInteger
    num_workers: NonNegativeInteger


class Seq2SeqConfig(StrictConfigModel):
    """Seq2Seq LSTM architecture configuration."""

    name: Literal["seq2seq_lstm"]
    embedding_dim: PositiveInteger
    hidden_dim: PositiveInteger
    num_layers: PositiveInteger
    dropout: DropoutRate


class BahdanauAttentionConfig(StrictConfigModel):
    """Seq2Seq with Bahdanau attention architecture configuration."""

    name: Literal["bahdanau_attention"]
    embedding_dim: PositiveInteger
    encoder_hidden_dim: PositiveInteger
    decoder_hidden_dim: PositiveInteger
    attention_dim: PositiveInteger
    num_layers: PositiveInteger
    dropout: DropoutRate


class TransformerConfig(StrictConfigModel):
    """Small Transformer architecture configuration."""

    name: Literal["transformer_small"]
    d_model: PositiveInteger
    num_heads: PositiveInteger
    num_encoder_layers: PositiveInteger
    num_decoder_layers: PositiveInteger
    feedforward_dim: PositiveInteger
    dropout: DropoutRate
    max_sequence_length: PositiveInteger

    @model_validator(mode="after")
    def validate_attention_dimensions(self) -> TransformerConfig:
        """Ensure that every attention head has the same integer dimension."""
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        return self


ModelConfig = Annotated[
    Seq2SeqConfig | BahdanauAttentionConfig | TransformerConfig,
    Field(discriminator="name"),
]


class TrainingConfig(StrictConfigModel):
    """Training configuration shared by the model scripts."""

    seed: NonNegativeInteger
    epochs: PositiveInteger
    learning_rate: PositiveFloat
    gradient_clip_norm: PositiveFloat | None
    evaluate_every: PositiveInteger
    checkpoint_directory: Path


class DataConfigFile(StrictConfigModel):
    """Schema of a data YAML file."""

    schema_version: Literal[1]
    data: DataConfig


class ModelConfigFile(StrictConfigModel):
    """Schema of a model YAML file."""

    schema_version: Literal[1]
    model: ModelConfig


class TrainingConfigFile(StrictConfigModel):
    """Schema of a training YAML file."""

    schema_version: Literal[1]
    training: TrainingConfig


class ExperimentConfig(StrictConfigModel):
    """Validated configuration used by one experiment."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


ConfigFileModel = TypeVar("ConfigFileModel", bound=BaseModel)
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser()

    try:
        with path.open(encoding="utf-8") as stream:
            content = yaml.safe_load(stream)
    except FileNotFoundError as error:
        raise ConfigError(f"Configuration file not found: {path}") from error
    except OSError as error:
        raise ConfigError(f"Cannot read configuration file {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {path}: {error}") from error

    if not isinstance(content, dict):
        raise ConfigError(f"Configuration root must be a mapping: {path}")

    return content


def _validate_file(
    config_path: str | Path,
    schema: type[ConfigFileModel],
) -> ConfigFileModel:
    try:
        return schema.model_validate(_read_yaml(config_path))
    except ValidationError as error:
        raise ConfigError(f"Invalid configuration in {config_path}:\n{error}") from error


def _resolve_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _resolve_project_root(project_root: str | Path | None) -> Path:
    if project_root is None:
        return DEFAULT_PROJECT_ROOT
    return Path(project_root).expanduser().resolve()


def load_data_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> DataConfig:
    """Load a data config and resolve its paths from the project root."""
    root = _resolve_project_root(project_root)
    parsed = _validate_file(config_path, DataConfigFile).data

    return parsed.model_copy(
        update={
            "train_path": _resolve_path(parsed.train_path, root),
            "validation_path": _resolve_path(parsed.validation_path, root),
            "test_path": _resolve_path(parsed.test_path, root),
        }
    )


def load_model_config(config_path: str | Path) -> ModelConfig:
    """Load a model config and select its schema from the model name."""
    return _validate_file(config_path, ModelConfigFile).model


def load_training_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> TrainingConfig:
    """Load a training config and resolve its output directory."""
    root = _resolve_project_root(project_root)
    parsed = _validate_file(config_path, TrainingConfigFile).training

    return parsed.model_copy(
        update={
            "checkpoint_directory": _resolve_path(
                parsed.checkpoint_directory,
                root,
            )
        }
    )


def load_config(
    data_config_path: str | Path,
    model_config_path: str | Path,
    training_config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ExperimentConfig:
    """Load the complete validated configuration for one experiment."""
    root = _resolve_project_root(project_root)

    return ExperimentConfig(
        data=load_data_config(data_config_path, project_root=root),
        model=load_model_config(model_config_path),
        training=load_training_config(training_config_path, project_root=root),
    )
