"""Translate one English sentence with the trained Seq2Seq model."""

import argparse

import torch

from machine_translation import (
    build_model,
    load_checkpoint,
    load_config,
    load_experiment,
    load_tokenizer,
    translate,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Translate English to French.")
    parser.add_argument("text", nargs="+", help="English text to translate")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--config",
        help="experiment YAML (default: configs/seq2seq_tatoeba.yaml)",
    )
    source.add_argument(
        "--run-dir",
        help="saved experiment directory containing config.yaml and checkpoints/",
    )
    parser.add_argument("--seed", type=int, help="run seed to load")
    parser.add_argument("--checkpoint", help="checkpoint file or directory")
    parser.add_argument("--checkpoint-kind", choices=("best", "last"), default="best")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    if args.run_dir and (args.seed is not None or args.checkpoint is not None):
        parser.error("--run-dir cannot be combined with --seed or --checkpoint")
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    if args.run_dir or (
        args.checkpoint and args.config is None and args.seed is None
    ):
        loaded = load_experiment(
            args.run_dir or args.checkpoint,
            device=device,
            checkpoint=args.checkpoint_kind,
        )
        config = loaded.config
        model = loaded.model
        source_tokenizer = loaded.source_tokenizer
        target_tokenizer = loaded.target_tokenizer
    else:
        config = load_config(args.config, seed=args.seed)
        tokens = config.tokenizer
        checkpoint = (
            args.checkpoint or config.checkpoint_directory / f"{args.checkpoint_kind}.pt"
        )

        source_tokenizer = load_tokenizer(
            tokens.source_artifact_path,
            tokens.special_tokens,
        )
        target_tokenizer = load_tokenizer(
            tokens.target_artifact_path,
            tokens.special_tokens,
        )
        model = build_model(
            source_tokenizer,
            target_tokenizer,
            tokens.special_tokens,
            config.model,
        )
        load_checkpoint(
            checkpoint, model, device, config=config, checkpoint=args.checkpoint_kind
        )
    tokens = config.tokenizer

    result = translate(
        model=model,
        source_text=" ".join(args.text),
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        tokenizer_config=tokens,
        data_config=config.data,
        device=device,
        beam_size=args.beam_size,
    )
    print(result)


if __name__ == "__main__":
    main()
