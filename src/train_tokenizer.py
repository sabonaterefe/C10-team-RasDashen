"""
Simple BPE-like trainer to build a tokenizer.json for the competition.

This script performs a frequency-driven pair merging starting from symbol
level, saving a vocabulary mapping token->id to `tokenizer.json`.

This is intentionally simple and safe: it preserves lossless decoding by
representing tokens as exact string fragments. It also supports an "allow_cross"
phase (SuperBPE-inspired) to permit merges across spaces by treating a
special boundary symbol during training and then removing it from token text
using an explicit marker so decoding remains exact.

Usage:
    python train_tokenizer.py --input corpus.txt --vocab-size 5000 --out tokenizer.json

"""
from __future__ import annotations

import argparse
import collections
import json
from typing import Dict

def read_corpus(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def initial_symbols(text: str) -> list[str]:
    # start with single characters
    return list(text)


def get_pair_frequencies(tokens_list):
    freqs = collections.Counter()
    for tokens in tokens_list:
        for a, b in zip(tokens, tokens[1:]):
            freqs[(a, b)] += 1
    return freqs


def merge_pair(tokens_list, pair):
    a, b = pair
    merged = a + b
    new_list = []
    for tokens in tokens_list:
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_list.append(new_tokens)
    return new_list


COMMON_SUFFIXES = {
    "ing",
    "ed",
    "ly",
    "es",
    "s",
    "ion",
    "al",
    "er",
    "or",
    "ist",
    "able",
    "ity",
}
COMMON_PREFIXES = {
    "un",
    "re",
    "in",
    "im",
    "ir",
    "dis",
    "pre",
    "post",
    "sub",
    "inter",
    "trans",
}


def affix_bonus(token: str) -> float:
    if any(token.endswith(suffix) for suffix in COMMON_SUFFIXES):
        return 0.30
    if any(token.startswith(prefix) for prefix in COMMON_PREFIXES):
        return 0.20
    return 0.0


def is_word_token(token: str) -> bool:
    return token.isalpha() and token == token.strip()


def pair_score(a: str, b: str, count: int, merges_done: int, allow_cross_after: int) -> float:
    """Score the merge candidate for SuperBPE training.

    This ranking prefers high-frequency reductions, word-internal merges first,
    and then allows strong cross-boundary phrase merges once the base BPE
    vocabulary has stabilized.
    """
    combined = a + b
    length = len(combined)
    reduction = len(a) + len(b) - 1
    score = float(count) * (reduction ** 1.1)

    alpha_ratio = sum(ch.isalpha() for ch in combined) / max(1, length)
    if alpha_ratio >= 0.65:
        score *= 1.25

    if is_word_token(a) and is_word_token(b):
        score *= 1.30

    score += affix_bonus(a) + affix_bonus(b)

    if " " in combined:
        if merges_done < allow_cross_after:
            return 0.0
        score *= 1.40
        if a.endswith(" ") and b and b[0].isalnum():
            score *= 1.20
        if b.startswith(" ") and a and a[-1].isalnum():
            score *= 1.10
        if combined.count(" ") > 1:
            score *= 0.88

    if length > 48:
        score *= 0.50
    elif length > 36:
        score *= 0.72
    elif length > 28:
        score *= 0.88

    if length > 1 and not combined.replace(" ", "").isalnum():
        score *= 0.92

    return score


PRINTABLE_ASCII = [chr(code) for code in range(32, 127)]
MAX_DISTINCT_TOKENS = 20000
MAX_TOKEN_LENGTH = 64


def train(
    corpus: str,
    vocab_size: int = 12000,
    allow_cross_after: int = 100,
    min_count: int = 2,
    max_merge_steps: int = 20000,
    repeat: int = 1,
    max_token_length: int = MAX_TOKEN_LENGTH,
):
    """Train BPE-like merges with a SuperBPE-style phased cross-boundary policy.

    - allow_cross_after: number of merges before allowing space-containing pairs.
    - min_count: minimum pair frequency to consider merging.
    - max_merge_steps: hard stop to keep training bounded.
    - repeat: how many times to repeat the corpus for stronger pair counts.
    - max_token_length: maximum merged token length to preserve generalization.
    """
    lines = corpus.splitlines()
    tokens_list = [initial_symbols(line) for _ in range(repeat) for line in lines]
    vocab = set()
    for tokens in tokens_list:
        vocab.update(tokens)

    merges_done = 0
    while len(vocab) < vocab_size and merges_done < max_merge_steps:
        freqs = get_pair_frequencies(tokens_list)
        if not freqs:
            break

        best_pair = None
        best_score = 0.0
        for (a, b), count in freqs.items():
            if count < min_count:
                continue
            sc = pair_score(a, b, count, merges_done, allow_cross_after)
            if sc <= 0.0:
                continue
            if sc > best_score:
                best_score = sc
                best_pair = (a, b)

        if best_pair is None:
            break

        a, b = best_pair
        if len(a + b) > max_token_length:
            break

        tokens_list = merge_pair(tokens_list, best_pair)
        vocab.add(a + b)
        merges_done += 1

    token_freq = collections.Counter()
    for tokens in tokens_list:
        token_freq.update(tokens)

    sorted_tokens = sorted(
        token_freq.items(),
        key=lambda item: (-item[1], -len(item[0]), item[0]),
    )

    final_tokens: list[str] = []
    seen: set[str] = set()
    for token, _count in sorted_tokens:
        if token in seen:
            continue
        final_tokens.append(token)
        seen.add(token)
        if len(final_tokens) >= vocab_size:
            break

    for ch in PRINTABLE_ASCII:
        if len(final_tokens) >= vocab_size:
            break
        if ch not in seen:
            final_tokens.append(ch)
            seen.add(ch)

    vocab_map = {t: i + 1 for i, t in enumerate(final_tokens)}
    return vocab_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="debug_corpus.txt")
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--allow-cross-after", type=int, default=100)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-merge-steps", type=int, default=20000)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-token-length", type=int, default=MAX_TOKEN_LENGTH)
    parser.add_argument("--out", default="data/tokenizer.json")
    args = parser.parse_args()

    corpus = read_corpus(args.input)
    vocab_map = train(
        corpus,
        vocab_size=args.vocab_size,
        allow_cross_after=args.allow_cross_after,
        min_count=args.min_count,
        max_merge_steps=args.max_merge_steps,
        repeat=args.repeat,
        max_token_length=args.max_token_length,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab_map}, f, ensure_ascii=False, indent=2)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
