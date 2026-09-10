"""Load the data and tokenizer YAML configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


PositiveInteger = Annotated[int, Field(gt=0)]
ConfigType = TypeVar("ConfigType", bound=BaseModel)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class DataConfig(ConfigModel):
    name: str
    source_language: str
    target_language: str
    train_path: Path
    validation_path: Path
    test_path: Path
    max_sequence_length: PositiveInteger
    batch_size: PositiveInteger
    num_workers: Annotated[int, Field(ge=0)]


class SpecialTokensConfig(ConfigModel):
    pad: str
    unk: str
    bos: str
    eos: str

    @model_validator(mode="after")
    def validate_unique_tokens(self) -> SpecialTokensConfig:
        if len(set(self.as_tuple())) != 4:
            raise ValueError("pad, unk, bos and eos tokens must be distinct")
        return self

    def as_tuple(self) -> tuple[str, str, str, str]:
        return self.pad, self.unk, self.bos, self.eos


class TokenizerConfig(ConfigModel):
    name: Literal["shared_unigram"]
    vocab_size: PositiveInteger
    source_column: str
    target_column: str
    deduplicate: bool
    artifact_path: Path
    special_tokens: SpecialTokensConfig

    @model_validator(mode="after")
    def validate_settings(self) -> TokenizerConfig:
        if self.source_column == self.target_column:
            raise ValueError("source and target columns must be distinct")
        if self.vocab_size <= 4:
            raise ValueError("vocab_size must be greater than 4")
        return self


def _load_section(
    path: str | Path,
    section: str,
    model: type[ConfigType],
) -> ConfigType:
    config_path = Path(path)
    try:
        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"Cannot load {config_path}: {error}") from error

    if not isinstance(content, dict) or content.get("schema_version") != 1:
        raise ConfigError(f"Invalid schema in {config_path}")
    if section not in content:
        raise ConfigError(f"Missing '{section}' section in {config_path}")

    try:
        return model.model_validate(content[section])
    except ValidationError as error:
        raise ConfigError(f"Invalid {section} config in {config_path}:\n{error}") from error


def _resolve(path: Path, project_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_data_config(
    path: str | Path,
    project_root: str | Path | None = None,
) -> DataConfig:
    root = Path(project_root).resolve() if project_root else PROJECT_ROOT
    config = _load_section(path, "data", DataConfig)
    return config.model_copy(
        update={
            name: _resolve(getattr(config, name), root)
            for name in ("train_path", "validation_path", "test_path")
        }
    )


def load_tokenizer_config(
    path: str | Path,
    project_root: str | Path | None = None,
) -> TokenizerConfig:
    root = Path(project_root).resolve() if project_root else PROJECT_ROOT
    config = _load_section(path, "tokenizer", TokenizerConfig)
    return config.model_copy(
        update={"artifact_path": _resolve(config.artifact_path, root)}
    )
