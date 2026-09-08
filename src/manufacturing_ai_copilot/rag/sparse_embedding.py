from collections import Counter
import hashlib
import re

import jieba
from qdrant_client import models

VALID_TOKEN_PATTERN = re.compile(r"[a-z0-9\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    normalized_text = text.casefold()
    raw_tokens = jieba.lcut(normalized_text)

    tokens = []

    for token in raw_tokens:
        token = token.strip()

        if token and VALID_TOKEN_PATTERN.search(token):
            tokens.append(token)
    return tokens


def token_to_index(token: str) -> int:
    token_bytes = token.encode("utf-8")

    digest = hashlib.blake2b(
        token_bytes,
        digest_size=4,
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="big",
        signed=False,
    )


def encode_sparse(text: str) -> models.SparseVector:
    token_counts = Counter(tokenize(text))
    weights: dict[int, float] = {}

    for token, count in token_counts.items():
        index = token_to_index(token)
        weights[index] = weights.get(index, 0.0) + float(count)

    ordered_weights = sorted(weights.items())

    return models.SparseVector(
        indices=[index for index, _ in ordered_weights],
        values=[value for _, value in ordered_weights],
    )
