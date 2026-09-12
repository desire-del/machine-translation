import json
import math
from time import perf_counter

import torch
import yaml
from einops import rearrange
from torch import nn
from tqdm.auto import tqdm

from machine_translation.config import Config
from machine_translation.data import get_data_loaders


def _save_run_files(config, history, device, model):
    config.run_directory.mkdir(parents=True, exist_ok=True)
    config_path = config.run_directory / "config.yaml"
    history_path = config.run_directory / "history.json"
    metadata_path = config.run_directory / "metadata.json"

    config_path.write_text(
        yaml.safe_dump(config.snapshot(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    history_path.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "pytorch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    if device.type == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name(device)
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    evaluated = [
        (epoch, loss)
        for epoch, loss in enumerate(history["val_loss"], start=1)
        if loss is not None
    ]
    if evaluated:
        best_epoch, best_loss = min(evaluated, key=lambda item: item[1])
        metrics = {
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "best_val_perplexity": history["val_perplexity"][best_epoch - 1],
            "training_seconds": sum(
                value for value in history["epoch_time"] if value is not None
            ),
        }
        (config.run_directory / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )


def _validate_resume(checkpoint, config):
    saved = checkpoint.get("config")
    if saved is None:
        return

    current = config.snapshot()
    for key in ("name", "seed", "data", "tokenizer", "model"):
        if saved[key] != current[key]:
            raise ValueError(f"Checkpoint and current config differ in '{key}'")


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    history,
    best_val_loss,
    config,
    epochs_without_improvement,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "config": config.snapshot(),
        },
        path,
    )


def load_checkpoint(path, model, device, optimizer=None, config=None):
    """Load model weights and, when provided, the optimizer state."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if config is not None:
        _validate_resume(checkpoint, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    teacher_forcing_ratio=0.5,
    gradient_clip_norm=None,
):
    model.train()
    running_loss = 0.0
    total_tokens = 0

    batch_bar = tqdm(
        dataloader,
        desc="Steps",
        position=1,
        leave=False,
    )

    for batch in batch_bar:
        optimizer.zero_grad()

        source_ids = batch["source_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        source_lengths = batch["source_lengths"]

        logits = model(
            source_ids,
            source_lengths,
            target_ids,
            teacher_forcing_ratio,
        )
        labels = target_ids[:, 1:]

        loss = criterion(
            rearrange(logits, "batch time vocab -> (batch time) vocab"),
            rearrange(labels, "batch time -> (batch time)"),
        )
        loss.backward()

        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        optimizer.step()

        tokens = labels.ne(criterion.ignore_index).sum().item()
        running_loss += loss.item() * tokens
        total_tokens += tokens
        batch_bar.set_postfix(loss=running_loss / total_tokens)

    return running_loss / total_tokens


def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            source_ids = batch["source_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            source_lengths = batch["source_lengths"]

            logits = model(
                source_ids,
                source_lengths,
                target_ids,
                teacher_forcing_ratio=1.0,
            )

            labels = target_ids[:, 1:]

            loss = criterion(
                rearrange(logits, "batch time vocab -> (batch time) vocab"),
                rearrange(labels, "batch time -> (batch time)"),
            )

            tokens = labels.ne(criterion.ignore_index).sum().item()
            running_loss += loss.item() * tokens
            total_tokens += tokens

    return running_loss / total_tokens


def train_model(
    model,
    target_tokenizer,
    source_tokenizer,
    config: Config,
    device,
    resume_from=None,
    save=True,
):
    """Train with one experiment config and optionally resume a checkpoint."""
    existing_checkpoints = list(config.checkpoint_directory.glob("*.pt"))
    if save and resume_from is None and existing_checkpoints:
        raise FileExistsError(
            f"Run already exists: {config.run_directory}. "
            "Use --resume or choose another --seed."
        )

    data_config = config.data
    train_config = config.training
    special_tokens = config.tokenizer.special_tokens
    target_pad_id = target_tokenizer.token_to_id(special_tokens.pad)

    train_loader, validation_loader, _ = get_data_loaders(
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        data_config=data_config,
        special_tokens=special_tokens,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_perplexity": [],
        "epoch_time": [],
    }
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=target_pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if resume_from is not None:
        checkpoint = load_checkpoint(
            path=resume_from,
            model=model,
            device=device,
            optimizer=optimizer,
            config=config,
        )
        start_epoch = checkpoint["epoch"]
        history = checkpoint["history"]
        best_val_loss = checkpoint["best_val_loss"]
        epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement",
            0,
        )

    history.setdefault("val_perplexity", [None] * len(history["train_loss"]))
    history.setdefault("epoch_time", [None] * len(history["train_loss"]))
    if save:
        _save_run_files(config, history, device, model)

    epoch_bar = tqdm(
        range(start_epoch, train_config.epochs),
        desc="Epochs",
        position=0,
        leave=True,
    )

    for epoch in epoch_bar:
        epoch_start = perf_counter()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            teacher_forcing_ratio=train_config.teacher_forcing_ratio,
            gradient_clip_norm=train_config.gradient_clip_norm,
        )
        history["train_loss"].append(train_loss)
        is_best = False

        should_evaluate = (
            (epoch + 1) % train_config.evaluate_every == 0
            or epoch + 1 == train_config.epochs
        )
        if should_evaluate:
            val_loss = evaluate(
                model=model,
                val_loader=validation_loader,
                criterion=criterion,
                device=device,
            )
            val_perplexity = math.exp(val_loss)
            history["val_loss"].append(val_loss)
            history["val_perplexity"].append(val_perplexity)

            if val_loss < best_val_loss - train_config.min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                is_best = True
            else:
                epochs_without_improvement += 1
        else:
            history["val_loss"].append(None)
            history["val_perplexity"].append(None)

        epoch_time = perf_counter() - epoch_start
        history["epoch_time"].append(epoch_time)

        if should_evaluate:
            epoch_bar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_perplexity=f"{val_perplexity:.2f}",
                epoch_time=f"{epoch_time:.1f}s",
            )
        else:
            epoch_bar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                epoch_time=f"{epoch_time:.1f}s",
            )

        print(f"Epoch {epoch + 1}/{train_config.epochs} completed in {epoch_time:.1f}s,train_loss={train_loss:.4f}, val_loss={history['val_loss'][-1]}, val_perplexity={history['val_perplexity'][-1]}")

        if save and is_best:
            save_checkpoint(
                config.checkpoint_directory / "best.pt",
                model,
                optimizer,
                epoch + 1,
                history,
                best_val_loss,
                config,
                epochs_without_improvement,
            )

        if save:
            save_checkpoint(
                config.checkpoint_directory / "last.pt",
                model,
                optimizer,
                epoch + 1,
                history,
                best_val_loss,
                config,
                epochs_without_improvement,
            )
            _save_run_files(config, history, device, model)

        should_stop = (
            should_evaluate
            and train_config.patience is not None
            and epochs_without_improvement >= train_config.patience
        )
        if should_stop:
            tqdm.write(
                f"Early stopping after {epoch + 1} epochs "
                f"(best validation loss: {best_val_loss:.4f})."
            )
            break

    return history
