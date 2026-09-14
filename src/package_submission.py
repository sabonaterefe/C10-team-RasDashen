"""
Create a submission ZIP containing `tokenizer.py` and `tokenizer.json`.

Usage:
    python src/package_submission.py --out submission.zip --model data/tokenizer.json
"""
import argparse
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submission.zip")
    parser.add_argument("--model", default="data/tokenizer.json")
    parser.add_argument("--entry", default="src/tokenizer.py")
    args = parser.parse_args()

    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(args.entry, arcname="tokenizer.py")
        z.write(args.model, arcname="tokenizer.json")
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
