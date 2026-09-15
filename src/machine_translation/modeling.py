"""Model construction from a validated experiment configuration."""

from machine_translation.seq2seq import build_seq2seq
from machine_translation.seq2seq_att import build_seq2seqatt


def build_model(
    source_tokenizer,
    target_tokenizer,
    special_tokens,
    model_config,
):
    """Build the architecture selected by ``model_config.type``."""
    builders = {
        "seq2seq": build_seq2seq,
        "seq2seq_att": build_seq2seqatt,
    }

    try:
        builder = builders[model_config.type]
    except KeyError as error:
        raise ValueError(
            f"Unsupported model type: {model_config.type!r}"
        ) from error

    return builder(
        source_tokenizer,
        target_tokenizer,
        special_tokens,
        model_config,
    )
