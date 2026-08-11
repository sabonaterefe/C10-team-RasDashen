# Build a SuperBPE Tokenizer — Repository

Professionalized starter kit for the SuperBPE tokenizer competition. This
repository contains a trainable, lossless, symbol-level tokenizer and
convenience scripts for training and evaluation.

Contents
- `tokenizer.py` — Required submission entrypoint (lossless encoder/decoder).
- `tokenizer.json` — Example trained vocabulary produced by `train_tokenizer.py`.
- `train_tokenizer.py` — Simple BPE-like trainer (development/debug use).
- `evaluate.py` — Small evaluation script for round-trip testing and stats.
- `preporocess_text.py` — Reference preprocessing used by the challenge.
- `debug_corpus.txt` — Small debug corpus for local testing.
- `tests/` — Unit tests validating round-trip correctness.

Quick start
1. Train a tokenizer (development):

```bash
python train_tokenizer.py --input debug_corpus.txt --vocab-size 5000 --out tokenizer.json
```

2. Verify encode/decode round-trip and stats:

```bash
python evaluate.py --model tokenizer.json --input debug_corpus.txt
```

3. Run unit tests:

```bash
python -m unittest discover -v
```

Design notes
- The submission entrypoint must be the file `tokenizer.py` with a `Tokenizer`
	class implementing `__init__`, `encode`, and `decode` methods. The evaluator
	instantiates separate encoder and decoder objects, therefore all learned
	assets must be serialized to disk (e.g., `tokenizer.json`) and loaded from
	`__init__`.
- The implementation in this repository is intentionally simple and
	trainable. Use it as a starting point to implement the SuperBPE ideas:
	frequency-driven merges that are allowed to cross whitespace after a
	configurable phase, while preserving lossless decoding.

Packaging for submission
- The competition expects a single archive containing `tokenizer.py` at the
	archive root. Optionally include `tokenizer.json` (learned vocab) and any
	small helper scripts, but avoid network access or external dependencies.

Submission checklist
- Ensure `tokenizer.py` implements the `Tokenizer` class and loads any
	required `tokenizer.json` assets in `__init__`.
- Keep the number of distinct token IDs <= 20,000.
- Verify lossless round-trip on the debug and development corpora.
- Measure end-to-end runtime (init + encode + decode) and keep it under
	the 20-minute limit for the 2,000,000-character final corpus.

Packaging helper
- `package_submission.py` creates `submission.zip` containing `tokenizer.py`
	and `tokenizer.json` ready for upload.

Benchmarking
- `benchmark.py` repeats a small corpus to approximate the final corpus size
	and measures init/encode/decode times. Use this locally to estimate runtime
	on the competition platform.

License & terms
By participating, you agree to submit only code and tokenizer assets that you
are permitted to use and distribute. Do not include private data, credentials,
malware, network-dependent code, or attempts to interfere with the evaluator.
Submissions are used solely to encode and decode the provided task corpus. Do
not include third-party assets with incompatible licenses.

Contact
Use issues in this repository for development questions and debugging.
