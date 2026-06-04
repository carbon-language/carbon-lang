#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# ///


"""Renames proposal files to match convention."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import os
import pathlib
import re
import sys
from typing import List, Optional, Tuple

# Add repo root to sys.path to allow importing from proposals.scripts
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from proposals.scripts.utils import slugify  # noqa: E402


def _parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        metavar="FILE",
        nargs="*",
        type=pathlib.Path,
        default=None,
        help="Specific proposals to check. If omitted, checks all proposals.",
    )
    return parser.parse_args(args=args)


# TODO: Switch to `pathspec` once it is available as a dependency to
# correctly parse `.gitignore` and allow an unfiltered walk of the repo.
def _find_markdown_files(repo_root: pathlib.Path) -> List[pathlib.Path]:
    """Finds all markdown files to update in our repository."""
    md_files = []
    # Top-level files
    for f in repo_root.glob("*.md"):
        md_files.append(f)

    # Top-level directories that contain markdown. We do these independently to
    # avoid having to filter out ignored directories.
    for dir_name in ["docs", "proposals", "toolchain"]:
        dir_path = repo_root / dir_name
        if dir_path.is_dir():
            md_files.extend(dir_path.glob("**/*.md"))

    return md_files


def _update_references(
    repo_root: pathlib.Path, renames: List[Tuple[str, str]]
) -> None:
    """Updates references to renamed proposals in all markdown files."""
    md_files = _find_markdown_files(repo_root)
    for md_file in md_files:
        with open(md_file, "r") as f:
            content = f.read()

        updated_content = content
        # Sort renames by length descending to avoid partial matches.
        for old_name, new_name in sorted(
            renames, key=lambda x: len(x[0]), reverse=True
        ):
            if old_name.endswith(".md"):
                # File reference: replace stand-alone occurrences.
                pattern = r"\b" + re.escape(old_name) + r"\b"
                updated_content = re.sub(pattern, new_name, updated_content)
            else:
                # Directory reference: only if part of a path or has anchor.
                # Case 1: followed by / or #
                pattern1 = r"\b" + re.escape(old_name) + r"(?=[/#])"
                updated_content = re.sub(pattern1, new_name, updated_content)

                # Case 2: preceded by /
                pattern2 = r"(?<=/)" + re.escape(old_name) + r"\b"
                updated_content = re.sub(pattern2, new_name, updated_content)

        if updated_content != content:
            print(f"Updating references in {md_file}")
            with open(md_file, "w") as f:
                f.write(updated_content)


def _rename_proposals(
    proposals_dir: pathlib.Path, files: Optional[List[pathlib.Path]] = None
) -> None:
    """Renames proposal files to match convention."""
    if not files:
        files = list(proposals_dir.glob("p*.md"))

    renames = []
    for path in files:
        filename = path.name

        # Make sure the filename starts with `p` followed by digits, and extract
        # the proposal number from those digits.
        match = re.match(r"p(\d+)", filename)
        if not match:
            print(f"Skipping non-conforming file: {filename}")
            continue
        num_str = match.group(1)
        num = int(num_str)

        # Read title from file
        title = None
        with open(path, "r") as f:
            for line in f:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

        if not title:
            print(f"Warning: No title found in {filename}")
            title = "untitled"

        slug = slugify(title)
        new_filename = f"p{num:06d}-{slug}.md"

        if filename != new_filename:
            renames.append((filename, new_filename))
            new_path = proposals_dir / new_filename
            print(f"Renaming {filename} to {new_filename}")
            os.rename(path, new_path)

        # Find any associated directory by matching the old filename without
        # the `.md` extensions, or directory named with `p` and the number.
        old_dir_path = None
        for dir in (f for f in proposals_dir.iterdir() if f.is_dir()):
            match_dir = re.match(r"^p(\d+)$", dir.name)
            if dir.name == filename[:-3] or (
                match_dir and int(match_dir.group(1)) == num
            ):
                old_dir_path = dir
                break

        if old_dir_path:
            new_dir_path = proposals_dir / new_filename[:-3]
            if old_dir_path != new_dir_path:
                renames.append((old_dir_path.name, new_dir_path.name))
                print(f"Renaming directory {old_dir_path} to {new_dir_path}")
                os.rename(old_dir_path, new_dir_path)

    if renames:
        repo_root = proposals_dir.absolute().parent
        _update_references(repo_root, renames)


def main(argv: Optional[List[str]] = None) -> None:
    proposals_dir = pathlib.Path("proposals")
    if not proposals_dir.is_dir():
        print("Error: proposals directory not found.")
        sys.exit(1)

    args = _parse_args(argv)
    _rename_proposals(proposals_dir, args.files)


if __name__ == "__main__":
    main()
