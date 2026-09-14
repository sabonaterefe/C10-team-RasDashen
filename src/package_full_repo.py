"""
Create a ZIP archive of the full project workspace for sharing or backup.

Usage:
    python src/package_full_repo.py --out project_bundle.zip

Excludes common directories like __pycache__, .git, .vscode, and .env files.
"""
import argparse
import os
import zipfile

EXCLUDE_DIRS = {"__pycache__", ".git", ".vscode", ".venv"}
EXCLUDE_FILES_SUFFIX = {".pyc", ".pyo"}


def should_include(path: str) -> bool:
    parts = set(path.split(os.sep))
    if parts & EXCLUDE_DIRS:
        return False
    if any(path.endswith(suf) for suf in EXCLUDE_FILES_SUFFIX):
        return False
    if os.path.basename(path).startswith(".") and os.path.basename(path) not in (".gitignore",):
        return False
    return True


def make_zip(root: str, out_path: str):
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root)
                if should_include(rel):
                    z.write(full, arcname=rel)
    print("Wrote", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="project_bundle.zip")
    args = parser.parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    make_zip(root, args.out)
