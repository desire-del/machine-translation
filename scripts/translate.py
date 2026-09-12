"""Translate one English sentence with the trained Seq2Seq model."""

import argparse

import torch

from machine_translation import (
    build_seq2seq,
    load_checkpoint,
    load_config,
    load_tokenizer,
    translate,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Translate English to French.")
    parser.add_argument("text", nargs="+", help="English text to translate")
    parser.add_argument(
        "--config",
        help="experiment YAML (default: configs/seq2seq_tatoeba.yaml)",
    )
    parser.add_argument("--seed", type=int, help="run seed to load")
    parser.add_argument("--checkpoint")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    config = load_config(args.config, seed=args.seed)
    tokens = config.tokenizer
    checkpoint = (
        args.checkpoint or config.checkpoint_directory / "best.pt"
    )

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
    load_checkpoint(checkpoint, model, device, config=config)

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
