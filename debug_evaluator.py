"""
Debug evaluator for tokenizer development.

Performs detailed diagnostics and writes `diagnostics.json`. On failure attempts
to set `total_tokens: -1` in the diagnostics to signal a failed run for scoring.

Usage:
    python debug_evaluator.py --model tokenizer.json --input debug_corpus.txt
"""
from __future__ import annotations

import argparse
import json
import traceback
from typing import Any


def escape_trunc(s: str, max_len: int = 200) -> str:
    escaped = s.encode("unicode_escape").decode("ascii")
    if len(escaped) > max_len:
        return escaped[: max_len - 3] + "..."
    return escaped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tokenizer.json")
    parser.add_argument("--input", default="debug_corpus.txt")
    args = parser.parse_args()

    diagnostics: dict[str, Any] = {
        "ok": False,
        "error": None,
        "traceback": None,
        "total_tokens": None,
        "distinct_tokens": None,
        "mismatches": [],
    }

    try:
        # read input
        with open(args.input, "r", encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f]

        # import tokenizer
        try:
            from tokenizer import Tokenizer
        except Exception as e:
            diagnostics["error"] = f"import_error: {e}"
            diagnostics["traceback"] = traceback.format_exc()
            diagnostics["total_tokens"] = -1
            print(json.dumps(diagnostics, indent=2))
            return

        # construct encoder
        try:
            encoder = Tokenizer(args.model)
        except Exception as e:
            diagnostics["error"] = f"construct_encoder_error: {e}"
            diagnostics["traceback"] = traceback.format_exc()
            diagnostics["total_tokens"] = -1
            print(json.dumps(diagnostics, indent=2))
            return

        # encode
        try:
            encoded = encoder.encode(lines)
        except Exception as e:
            diagnostics["error"] = f"encode_error: {e}"
            diagnostics["traceback"] = traceback.format_exc()
            diagnostics["total_tokens"] = -1
            print(json.dumps(diagnostics, indent=2))
            return

        # validate encoded batch type and lengths
        if not isinstance(encoded, list):
            diagnostics["error"] = "encoded_not_list"
            diagnostics["traceback"] = None
            diagnostics["total_tokens"] = -1
            print(json.dumps(diagnostics, indent=2))
            return
        if len(encoded) != len(lines):
            diagnostics["error"] = "encoded_length_mismatch"
            diagnostics["traceback"] = None
            diagnostics["total_tokens"] = -1
            print(json.dumps(diagnostics, indent=2))
            return

        total_tokens = 0
        distinct = set()
        for i, row in enumerate(encoded):
            if not isinstance(row, list):
                diagnostics["error"] = f"encoded_row_not_list_at_{i}"
                diagnostics["traceback"] = None
                diagnostics["total_tokens"] = -1
                print(json.dumps(diagnostics, indent=2))
                return
            for tok in row:
                if not isinstance(tok, int):
                    diagnostics["error"] = f"token_not_int_at_{i}: {repr(tok)}"
                    diagnostics["traceback"] = None
                    diagnostics["total_tokens"] = -1
                    print(json.dumps(diagnostics, indent=2))
                    return
                distinct.add(tok)
                total_tokens += 1

        diagnostics["total_tokens"] = total_tokens
        diagnostics["distinct_tokens"] = len(distinct)

        # check distinct token limit
        if diagnostics["distinct_tokens"] > 20000:
            diagnostics["error"] = "distinct_token_limit_exceeded"
            diagnostics["traceback"] = None
            diagnostics["total_tokens"] = -1
            print(json.dumps(diagnostics, indent=2))
            return

        # construct fresh decoder
        try:
            decoder = Tokenizer(args.model)
        except Exception as e:
            diagnostics["error"] = f"construct_decoder_error: {e}"
            diagnostics["traceback"] = traceback.format_exc()
            diagnostics["total_tokens"] = -1
            print(json.dumps(diagnostics, indent=2))
            return

        # decode
        try:
            decoded = decoder.decode(encoded)
        except Exception as e:
            diagnostics["error"] = f"decode_error: {e}"
            diagnostics["traceback"] = traceback.format_exc()
            diagnostics["total_tokens"] = -1
            print(json.dumps(diagnostics, indent=2))
            return

        # validate decoded
        if not isinstance(decoded, list):
            diagnostics["error"] = "decoded_not_list"
            diagnostics["traceback"] = None
            diagnostics["total_tokens"] = -1
            print(json.dumps(diagnostics, indent=2))
            return
        if len(decoded) != len(lines):
            diagnostics["error"] = "decoded_length_mismatch"
            diagnostics["traceback"] = None
            diagnostics["total_tokens"] = -1
            print(json.dumps(diagnostics, indent=2))
            return

        mismatches = []
        for i, (orig, out) in enumerate(zip(lines, decoded)):
            if not isinstance(out, str):
                diagnostics["error"] = f"decoded_value_not_string_at_{i}"
                diagnostics["traceback"] = None
                diagnostics["total_tokens"] = -1
                print(json.dumps(diagnostics, indent=2))
                return
            if orig != out:
                mismatches.append(
                    {
                        "index": i,
                        "orig": escape_trunc(orig),
                        "decoded": escape_trunc(out),
                    }
                )
                if len(mismatches) >= 10:
                    break

        diagnostics["mismatches"] = mismatches
        diagnostics["ok"] = len(mismatches) == 0
        if not diagnostics["ok"]:
            diagnostics["total_tokens"] = -1

    except Exception as e:
        diagnostics["error"] = f"unexpected_error: {e}"
        diagnostics["traceback"] = traceback.format_exc()
        diagnostics["total_tokens"] = -1

    # write diagnostics.json
    with open("diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
