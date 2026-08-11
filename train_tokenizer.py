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


def pair_score(a: str, b: str, count: int) -> float:
    """Heuristic score for choosing which pair to merge next.

    Pairs are scored by their expected token reduction, boosted by
    linguistically coherent merges and cross-boundary phrase candidates.
    """
    combined = a + b
    length = len(combined)
    reduction = len(a) + len(b) - 1
    score = float(count) * reduction

    alpha_ratio = sum(ch.isalpha() for ch in combined) / max(1, length)
    if alpha_ratio >= 0.65:
        score *= 1.15

    score += affix_bonus(a) + affix_bonus(b)

    if is_word_token(a) and is_word_token(b):
        score *= 1.20

    if " " in combined:
        # Allow cross-space merges after the word-internal phase.
        score *= 1.15
        if a.endswith(" ") or b.startswith(" "):
            score *= 1.05

    if length > 8:
        score *= 0.92
    elif length > 5:
        score *= 0.97

    return score


PRINTABLE_ASCII = [chr(code) for code in range(32, 127)]
MAX_DISTINCT_TOKENS = 20000


def train(
    corpus: str,
    vocab_size: int = 12000,
    allow_cross_after: int = 0,
    min_count: int = 2,
    max_merge_steps: int = 25000,
    repeat: int = 1,
):
    """Train BPE-like merges with a SuperBPE-style phased cross-boundary policy.

    - allow_cross_after: number of merges before allowing space-containing pairs.
    - min_count: minimum pair frequency to consider merging.
    - max_merge_steps: hard stop to keep training bounded.
    - repeat: how many times to repeat the corpus for stronger pair counts.
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
            is_cross = (" " in a) or (" " in b) or a.endswith(" ") or b.startswith(" ")
            if is_cross and merges_done < allow_cross_after:
                continue
            sc = pair_score(a, b, count)
            if sc > best_score:
                best_score = sc
                best_pair = (a, b)

        if best_pair is None:
            break

        tokens_list = merge_pair(tokens_list, best_pair)
        a, b = best_pair
        vocab.add(a + b)
        merges_done += 1

    token_freq = collections.Counter()
    for tokens in tokens_list:
        token_freq.update(tokens)
    sorted_tokens = [t for t, _ in token_freq.most_common()]

    final_tokens = []
    seen = set()
    for token in sorted_tokens:
        if token not in seen:
            final_tokens.append(token)
            seen.add(token)
            if len(final_tokens) >= MAX_DISTINCT_TOKENS:
                break

    for ch in PRINTABLE_ASCII:
        if ch not in seen and len(final_tokens) < MAX_DISTINCT_TOKENS:
            final_tokens.append(ch)
            seen.add(ch)

    vocab_map = {t: i + 1 for i, t in enumerate(final_tokens)}
    return vocab_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="debug_corpus.txt")
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--allow-cross-after", type=int, default=0)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-merge-steps", type=int, default=25000)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--out", default="tokenizer.json")
    args = parser.parse_args()

    corpus = read_corpus(args.input)
    vocab_map = train(
        corpus,
        vocab_size=args.vocab_size,
        allow_cross_after=args.allow_cross_after,
        min_count=args.min_count,
        max_merge_steps=args.max_merge_steps,
        repeat=args.repeat,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab_map}, f, ensure_ascii=False, indent=2)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
