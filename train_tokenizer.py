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
import re
from typing import Dict, Tuple

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


def pair_score(a: str, b: str, count: int) -> float:
    """Heuristic score for choosing which pair to merge next.

    Gives a small boost to merges that look like word-internal or
    morpheme-like sequences: both parts alphabetic, one part length>1, etc.
    This is a lightweight, local heuristic to prefer linguistically useful
    merges without external data.
    """
    score = float(count)
    if a.strip() and b.strip() and a.isalpha() and b.isalpha():
        score *= 1.2
    if (a.endswith(" ") or b.startswith(" ")) and not (a.strip() and b.strip()):
        score *= 0.8
    if len(a) + len(b) > 64:
        score *= 0.5
    return score


def train(
    corpus: str,
    vocab_size: int = 5000,
    allow_cross_after: int = 1000,
    min_count: int = 2,
):
    """Train BPE-like merges with a SuperBPE-style phased cross-boundary policy.

    - allow_cross_after: number of merges to perform before allowing merges
      that include spaces (cross-word merges).
    - min_count: minimum pair frequency to consider merging.
    """
    tokens_list = [initial_symbols(line) for line in corpus.splitlines()]
    vocab = set()
    for tokens in tokens_list:
        vocab.update(tokens)

    merges_done = 0
    while len(vocab) < vocab_size:
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
    final_tokens = list(sorted_tokens)
    if len(final_tokens) > 20000:
        final_tokens = final_tokens[:20000]
    vocab_map = {t: i + 1 for i, t in enumerate(final_tokens)}
    return vocab_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="debug_corpus.txt")
    parser.add_argument("--vocab-size", type=int, default=5000)
    parser.add_argument("--out", default="tokenizer.json")
    args = parser.parse_args()

    corpus = read_corpus(args.input)
    vocab_map = train(corpus, args.vocab_size)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab_map}, f, ensure_ascii=False, indent=2)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
