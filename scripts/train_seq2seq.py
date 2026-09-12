"""Train the Seq2Seq LSTM model."""

import argparse
import random

import numpy as np
import torch

from machine_translation import (
    build_seq2seq,
    load_config,
    load_tokenizer,
    train_model,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Seq2Seq model.")
    parser.add_argument(
        "--config",
        help="experiment YAML (default: configs/seq2seq_tatoeba.yaml)",
    )
    parser.add_argument("--seed", type=int, help="override the configured run seed")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="last",
        help="resume this run's last checkpoint or the given checkpoint",
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save the experiment (default: enabled)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, seed=args.seed)
    tokens = config.tokenizer
    resume_from = (
        config.checkpoint_directory / "last.pt"
        if args.resume == "last"
        else args.resume
    )

    set_seed(config.seed)
    device = torch.device(args.device)

    source_tokenizer = load_tokenizer(
        tokens.source_artifact_path,
        tokens.special_tokens,
    )
    target_tokenizer = load_tokenizer(
        tokens.target_artifact_path,
        tokens.special_tokens,
    )

    model = build_seq2seq(
        source_tokenizer,
        target_tokenizer,
        tokens.special_tokens,
        config.model,
    )

    print(f"Device: {device}")
    train_model(
        model=model,
        target_tokenizer=target_tokenizer,
        source_tokenizer=source_tokenizer,
        config=config,
        resume_from=resume_from,
        device=device,
        save=args.save,
    )
    if args.save:
        print(f"Experiment: {config.run_directory}")
    else:
        print("Experiment not saved.")


if __name__ == "__main__":
    main()
