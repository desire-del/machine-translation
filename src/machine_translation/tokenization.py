"""Build, save, load and use the shared translation tokenizer."""

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Encoding, Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import Unigram
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Metaspace as MetaspacePreTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import UnigramTrainer

from machine_translation.config import SpecialTokensConfig, TokenizerConfig


class TokenizationError(ValueError):
    """Raised when a tokenizer violates the expected project contract."""


def _validated_texts(texts: Iterable[str]) -> Iterable[str]:
    has_text = False

    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TokenizationError(f"Text at index {index} is not a string")
        if not text.strip():
            raise TokenizationError(f"Text at index {index} is empty")

        has_text = True
        yield text

    if not has_text:
        raise TokenizationError("Cannot train or encode an empty text collection")


def get_special_token_ids(
    tokenizer: Tokenizer,
    special_tokens: SpecialTokensConfig,
) -> dict[str, int]:
    """Resolve and validate the four required special-token IDs."""
    token_by_role = {
        "pad": special_tokens.pad,
        "unk": special_tokens.unk,
        "bos": special_tokens.bos,
        "eos": special_tokens.eos,
    }
    ids: dict[str, int] = {}

    for role, token in token_by_role.items():
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise TokenizationError(f"Missing {role} token in vocabulary: {token}")
        ids[role] = token_id

    if len(set(ids.values())) != len(ids):
        raise TokenizationError("Special tokens must have distinct vocabulary IDs")

    return ids


def build_tokenizer(
    texts: Iterable[str],
    config: TokenizerConfig,
) -> Tokenizer:
    """Build the configured shared Unigram tokenizer from raw texts."""
    tokenizer = Tokenizer(Unigram())
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = MetaspacePreTokenizer()
    tokenizer.decoder = MetaspaceDecoder()

    trainer = UnigramTrainer(
        vocab_size=config.vocab_size,
        unk_token=config.special_tokens.unk,
        special_tokens=list(config.special_tokens.as_tuple()),
    )

    tokenizer.train_from_iterator(_validated_texts(texts), trainer=trainer)

    special_ids = get_special_token_ids(tokenizer, config.special_tokens)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.special_tokens.bos} $A {config.special_tokens.eos}",
        special_tokens=[
            (config.special_tokens.bos, special_ids["bos"]),
            (config.special_tokens.eos, special_ids["eos"]),
        ],
    )

    return tokenizer


def save_tokenizer(
    tokenizer: Tokenizer,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a tokenizer without silently replacing an existing artifact."""
    artifact_path = Path(path).expanduser()

    if artifact_path.exists() and not overwrite:
        raise FileExistsError(f"Tokenizer artifact already exists: {artifact_path}")

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(artifact_path), pretty=True)
    return artifact_path.resolve()


def load_tokenizer(
    path: str | Path,
    special_tokens: SpecialTokensConfig,
) -> Tokenizer:
    """Load a tokenizer artifact and validate its required special tokens."""
    artifact_path = Path(path).expanduser()

    if not artifact_path.is_file():
        raise FileNotFoundError(f"Tokenizer artifact not found: {artifact_path}")

    tokenizer = Tokenizer.from_file(str(artifact_path))
    get_special_token_ids(tokenizer, special_tokens)
    return tokenizer


def encode_batch(
    tokenizer: Tokenizer,
    texts: Iterable[str],
) -> list[Encoding]:
    """Encode a non-empty batch with the tokenizer's serialized pipeline."""
    batch = list(_validated_texts(texts))
    return tokenizer.encode_batch(batch, add_special_tokens=True)
