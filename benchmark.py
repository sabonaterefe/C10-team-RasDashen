"""
Benchmark encode/decode speed by repeating a corpus to approx target characters.

Usage:
    python benchmark.py --model tokenizer.json --input debug_corpus.txt --target_chars 2000000

Note: This script is for local estimation. It will create a repeated corpus in memory
and measure timing for Tokenizer init, encode, and decode.
"""
from __future__ import annotations

import argparse
import time
from tokenizer import Tokenizer


def read_corpus(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def make_repeated_lines(corpus: str, target_chars: int):
    lines = corpus.splitlines()
    if not lines:
        return [""]
    out = []
    total = 0
    i = 0
    while total < target_chars:
        line = lines[i % len(lines)]
        out.append(line)
        total += len(line)
        i += 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tokenizer.json")
    parser.add_argument("--input", default="debug_corpus.txt")
    parser.add_argument("--target-chars", type=int, default=2000000)
    args = parser.parse_args()

    corpus = read_corpus(args.input)
    lines = make_repeated_lines(corpus, args.target_chars)

    t0 = time.time()
    enc = Tokenizer(args.model)
    t1 = time.time()
    encoded = enc.encode(lines)
    t2 = time.time()
    dec = Tokenizer(args.model)
    decoded = dec.decode(encoded)
    t3 = time.time()

    assert decoded == lines

    print(f"Lines: {len(lines)}, Target chars: {args.target_chars}")
    print(f"Init time: {t1-t0:.3f}s, Encode time: {t2-t1:.3f}s, Decode time: {t3-t2:.3f}s")


if __name__ == "__main__":
    main()
