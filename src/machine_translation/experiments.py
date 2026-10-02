"""Restore an experiment from its saved configuration and checkpoint."""

from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch import nn

from machine_translation.config import Config, load_config
from machine_translation.modeling import build_model
from machine_translation.tokenization import load_tokenizer
from machine_translation.training import load_checkpoint, resolve_checkpoint


@dataclass
class LoadedExperiment:
    model: nn.Module
    config: Config
    source_tokenizer: Tokenizer
    target_tokenizer: Tokenizer
    checkpoint_path: Path
    epoch: int | None


def load_experiment(path, device="cpu", checkpoint="best"):
    """Load a run directory, its checkpoints directory, or an exact .pt file.

    config.yaml must be beside the checkpoint or in its run directory when
    the checkpoint is under checkpoints/. Relative artifact paths retain the
    project-root convention used by saved configs. No checkpoint fallback occurs.
    """
    checkpoint_path = resolve_checkpoint(path, checkpoint=checkpoint)
    config_path = checkpoint_path.parent / "config.yaml"
    if not config_path.is_file() and checkpoint_path.parent.name == "checkpoints":
        config_path = checkpoint_path.parent.parent / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Saved experiment config not found: {config_path}")

    config = load_config(config_path)
    source_tokenizer = load_tokenizer(
        config.tokenizer.source_artifact_path,
        config.tokenizer.special_tokens,
    )
    target_tokenizer = load_tokenizer(
        config.tokenizer.target_artifact_path,
        config.tokenizer.special_tokens,
    )
    model = build_model(
        source_tokenizer,
        target_tokenizer,
        config.tokenizer.special_tokens,
        config.model,
    )
    # Optimizer tensors also occur in training checkpoints: keep them off the GPU.
    saved = load_checkpoint(checkpoint_path, model, "cpu", config=config)
    epoch = saved.get("epoch")
    del saved
    model.to(torch.device(device))
    model.eval()
    return LoadedExperiment(
        model=model,
        config=config,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        checkpoint_path=checkpoint_path,
        epoch=epoch,
    )
