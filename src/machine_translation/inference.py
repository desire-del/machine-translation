"""Autoregressive translation for the Seq2Seq model."""

import torch

from machine_translation.tokenization import get_special_token_ids


def translate(
    model,
    source_text,
    source_tokenizer,
    target_tokenizer,
    tokenizer_config,
    data_config,
    device,
    beam_size=5,
):
    """Translate one sentence with length-normalized beam search.

    Set ``beam_size=1`` to recover greedy decoding.
    """
    if not source_text.strip():
        raise ValueError("source_text cannot be empty")
    if beam_size < 1:
        raise ValueError("beam_size must be at least 1")
    if beam_size > target_tokenizer.get_vocab_size():
        raise ValueError("beam_size cannot exceed the target vocabulary size")

    model = model.to(device)
    model.eval()

    source_tokenizer.enable_truncation(
        max_length=data_config.max_sequence_length
    )
    source_ids = source_tokenizer.encode(
        source_text,
        add_special_tokens=True,
    ).ids
    source_ids = torch.tensor([source_ids], dtype=torch.long, device=device)
    source_lengths = torch.tensor([source_ids.size(1)])

    special_ids = get_special_token_ids(
        target_tokenizer,
        tokenizer_config.special_tokens,
    )

    with torch.inference_mode():
        hidden, cell = model.encoder(source_ids, source_lengths)
        beams = [([], 0.0, hidden, cell, False)]

        for _ in range(data_config.max_sequence_length - 1):
            candidates = []

            for token_ids, score, hidden, cell, finished in beams:
                if finished:
                    candidates.append((token_ids, score, hidden, cell, True))
                    continue

                input_id = token_ids[-1] if token_ids else special_ids["bos"]
                decoder_input = torch.tensor(
                    [[input_id]],
                    dtype=torch.long,
                    device=device,
                )
                logits, next_hidden, next_cell = model.decoder(
                    decoder_input,
                    hidden,
                    cell,
                )
                log_probs = logits[0, -1].log_softmax(dim=-1)
                log_probs[special_ids["pad"]] = -torch.inf
                log_probs[special_ids["bos"]] = -torch.inf

                values, indices = log_probs.topk(beam_size)
                for value, index in zip(values.tolist(), indices.tolist()):
                    candidates.append(
                        (
                            token_ids + [index],
                            score + value,
                            next_hidden,
                            next_cell,
                            index == special_ids["eos"],
                        )
                    )

            candidates.sort(
                key=lambda beam: beam[1]
                / (((5 + len(beam[0])) / 6) ** 0.6),
                reverse=True,
            )
            beams = candidates[:beam_size]

            if all(beam[-1] for beam in beams):
                break

    return target_tokenizer.decode(
        beams[0][0],
        skip_special_tokens=True,
    )
