# C10 SuperBPE Tokenizer

This repository contains a compact, lossless SuperBPE-inspired tokenizer implementation for the competition workflow. The code is organized so that the training pipeline, evaluation scripts, and submission packaging are reproducible from a clean checkout.

## Dataset

The repository includes a small development corpus for local debugging and verification:

- `debug_corpus.txt` — lightweight text corpus used for iterative development and testing.
- `data/tokenizer.json` — serialized tokenizer assets produced by the training pipeline.

For the full competition dataset, the repository expects the evaluator or organizer to provide the target corpus separately. The local scripts are designed to run with any UTF-8 text file in the same format.

## Training Pipeline

The tokenizer is trained with a BPE-style process that supports a SuperBPE-inspired phase where merges can cross whitespace boundaries after a configurable number of merge steps.

Main training-related files:

- `src/train_tokenizer.py` — training logic for building `tokenizer.json`.
- `src/tokenizer.py` — reusable tokenizer implementation.
- `src/package_submission.py` — helper for packaging submission artifacts.

Typical workflow:

1. Train a tokenizer:
   ```bash
   python src/train_tokenizer.py --input debug_corpus.txt --vocab-size 20000 --repeat 10 --out data/tokenizer.json
   ```
2. Inspect or verify the learned vocabulary.
3. Package the final model for submission when ready.

## Evaluation

The repository includes a lightweight evaluation flow to validate round-trip fidelity and basic token statistics.

Key evaluation files:

- `src/evaluate.py` — evaluates model quality and prints summary statistics.
- `tests/test_tokenizer_roundtrip.py` — end-to-end round-trip unit test.

Verification performed locally:

```bash
python -m unittest discover -s tests -v
```

This test confirms that encoded text can be decoded back to the original input without loss.

## Reproduction

Run the project in the following order:

1. Create or update the tokenizer model:
   ```bash
   python src/train_tokenizer.py --input debug_corpus.txt --vocab-size 20000 --repeat 10 --out data/tokenizer.json
   ```
2. Run the evaluation script:
   ```bash
   python src/evaluate.py --model data/tokenizer.json --input debug_corpus.txt
   ```
3. Run the unit test:
   ```bash
   python -m unittest discover -s tests -v
   ```
4. Build a submission archive:
   ```bash
   python src/package_submission.py --out submission.zip --model data/tokenizer.json
   ```

## Cohort Challenges

The repository includes the required cohort challenge materials in the `doc/` folder, including this README and supporting documentation for the submission workflow.

## Appendix

### Contributors / Team Members

- Sabona Terefe Bango
- other teams quitted at the start of the project

### Mentors

- Sabona Terefe Bango

## Repository Structure

- `src/` — core implementation and support scripts
- `tests/` — round-trip validation tests
- `data/` — packaged tokenizer assets
- `doc/` — submission documentation and requirements
- `debug_corpus.txt` — local development corpus
- `src/tokenizer.py` — core tokenizer implementation
- `src/train_tokenizer.py` — training pipeline entry point

## Notes

- The project is designed to run as a public GitHub repository with all reproducible assets included or clearly documented.
- The GitHub repository name should follow the competition convention `C10-team-name`.
- The top-level README is intentionally concise so it remains within the requested character limit while covering the required sections.
