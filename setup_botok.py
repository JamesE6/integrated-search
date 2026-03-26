#!/usr/bin/env python3
"""
setup_botok.py – Initialize botok with the Divergent Discourses custom dictionary.

This script automates the modern-botok setup process:
  1. Runs botok once to generate the default 'general' dialect pack
  2. Copies 'general' → 'custom'
  3. Replaces the dictionary with the bundled custom tsikchen.tsv
  4. Regenerates the trie (custom_trie.pickled)

Prerequisites:
  - botok==0.9.0 installed (via requirements.txt)
  - packages/modern-botok/ present in this repo (bundled)

Usage:
  python setup_botok.py [--base-path /custom/path/to/pybo]

Default base path: ./pybo (app-relative, not ~/Documents/pybo)
"""

import argparse
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Initialize botok custom dictionary")
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Where to create the pybo directory (default: ./pybo in repo root)",
    )
    args = parser.parse_args()

    # Resolve paths
    repo_root = Path(__file__).resolve().parent
    base_path = Path(args.base_path) if args.base_path else repo_root / "pybo"
    dialect_packs = base_path / "dialect_packs"

    custom_dict_src = repo_root / "packages" / "modern-botok" / "dictionary" / "tsikchen.tsv"
    if not custom_dict_src.exists():
        print(f"ERROR: Custom dictionary not found at {custom_dict_src}")
        print("Expected packages/modern-botok/dictionary/tsikchen.tsv")
        print("Ensure the modern-botok repo is present in packages/modern-botok/")
        sys.exit(1)

    # Step 1: Generate the default 'general' dialect pack
    print(f"Initializing botok at {base_path}…")
    dialect_packs.mkdir(parents=True, exist_ok=True)

    from botok.config import Config
    from botok import WordTokenizer

    config = Config(base_path=str(dialect_packs))
    wt = WordTokenizer(config=config)
    print("  ✓ Default 'general' dialect pack created")

    # Step 2: Copy general → custom
    general_dir = dialect_packs / "general"
    custom_dir = dialect_packs / "custom"

    if custom_dir.exists():
        print("  Removing existing 'custom' directory…")
        shutil.rmtree(custom_dir)

    shutil.copytree(general_dir, custom_dir)
    print("  ✓ Copied 'general' → 'custom'")

    # Step 3: Replace dictionary
    custom_dict_dest = custom_dir / "dictionary" / "words" / "tsikchen.tsv"
    custom_dict_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(custom_dict_src, custom_dict_dest)
    print("  ✓ Custom dictionary installed")

    # Step 4: Regenerate trie
    print("  Regenerating trie (this may take a moment)…")
    config = Config(dialect_name="custom", base_path=str(dialect_packs))
    wt = WordTokenizer(config=config)
    print("  ✓ Trie regenerated")

    print()
    print(f"Setup complete. Dialect packs: {dialect_packs}")
    print("The app will find this automatically — no path editing needed.")


if __name__ == "__main__":
    main()
