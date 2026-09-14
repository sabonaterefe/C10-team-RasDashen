"""
Evaluate tokenizer encode/decode roundtrip and basic compression stats.

Usage:
    python evaluate.py --model tokenizer.json --input debug_corpus.txt
"""
from __future__ import annotations

import argparse
import json
import time
from src.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/tokenizer.json")
    parser.add_argument("--input", default="debug_corpus.txt")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]

    t0 = time.time()
    enc = Tokenizer(args.model)
    t1 = time.time()
    encoded = enc.encode(lines)
    t2 = time.time()
    dec = Tokenizer(args.model)  # fresh decoder instance
    decoded = dec.decode(encoded)
    t3 = time.time()

    assert decoded == lines, "Round-trip failed"

    total_input_chars = sum(len(l) for l in lines)
    total_tokens = sum(len(seq) for seq in encoded)

    print(f"Input chars: {total_input_chars}")
    print(f"Emitted tokens: {total_tokens}")
    print(f"Encode time: {t2-t1:.4f}s, Decode time: {t3-t2:.4f}s, Init time: {(t1-t0)+(t3-t2):.4f}s")


if __name__ == "__main__":
    main()
