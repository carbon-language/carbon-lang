#!/usr/bin/env python3

"""Updates files in preparation for a jekyll build.

Used from .github/workflows/gh_pages.yaml. This updates the file and directory
structure prior to the jekyll build.
"""

import os
from pathlib import Path
import re
from typing import Optional

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""


def get_title(f: Path, content: str) -> str:
    """Returns a file's title according to markdown.

    Replacements are for YAML compatibility in `add_frontmatter`.
    """
    m = re.search("^# (.*)$", content, re.MULTILINE)
    assert m, f
    title = m[1]
    title = title.replace("\\", "\\\\")
    title = title.replace('"', '\\"')
    return title


def add_frontmatter(
    f: Path,
    orig_content: str,
    title: Optional[str],
    parent_title: Optional[str],
    nav_order: Optional[int],
    has_children: bool,
) -> None:
    """Adds frontmatter to a file."""
    content = "---\n"
    if title:
        content += f'title: "{title}"\n'
    if parent_title:
        content += f'parent: "{parent_title}"\n'
    if nav_order is not None:
        content += f"nav_order: {nav_order}\n"
    if has_children:
        # has_toc is disabling the list of child files in a separate table of
        # contents.
        content += "has_children: true\nhas_toc: false\n"
    content += "---\n\n" + orig_content
    f.write_text(content)


def generate_children(dir: Path, parent_title: Optional[str]) -> None:
    """Automatically adds child information to markdown files.

    This is in support of navigation.

    TODO: Handle nesting better. This is very so-so right now.
    """
    readme = None
    md_files = []
    subdirs = []

    for f in dir.iterdir():
        if f.is_file():
            if f.name == "README.md":
                readme = f
            elif f.suffix == ".md":
                md_files.append(f)
        elif f.is_dir():
            subdirs.append(f)

    if readme and md_files:
        readme_content = readme.read_text()
        readme_title = get_title(readme, readme_content)
        add_frontmatter(
            readme, readme_content, readme_title, parent_title, None, True
        )

        for md_file in md_files:
            content = md_file.read_text()
            md_title = get_title(md_file, content)
            nav_order = None

            # Use proposal numbers as part of the title and ordering.
            m = re.match(r"p(\d+).md", md_file.name)
            if m:
                md_title = f"#{m[1]}: {md_title}"
                nav_order = int(m[1])

            add_frontmatter(
                md_file, content, md_title, readme_title, nav_order, False
            )

    for subdir in subdirs:
        generate_children(subdir, None)


def label_root_file(name: str, title: str, nav_order: int) -> None:
    """Adds frontmatter to a root file, like CONTRIBUTING.md."""
    f = Path(name)
    add_frontmatter(f, f.read_text(), title, None, nav_order, False)


def main() -> None:
    # Ensure this runs from the repo root.
    os.chdir(Path(__file__).parent.parent)

    # bazel-execroot interferes with jekyll because it's a broken symlink.
    Path("bazel-execroot").unlink()

    # Move files to the repo root.
    for f in Path("website").iterdir():
        f.rename(f.name)

    # Add frontmatter to root files.
    label_root_file("README.md", "README", 0)
    label_root_file("CONTRIBUTING.md", "Contributing", 1)
    label_root_file("CODE_OF_CONDUCT.md", "Code of conduct", 2)
    label_root_file("SECURITY.md", "Security policy", 3)

    # Add frontmatter to child files.
    for subdir in Path(".").iterdir():
        if subdir.is_dir() and not (
            subdir.name == "external"
            or subdir.name.startswith(".")
            or subdir.name.startswith("bazel-")
        ):
            generate_children(subdir, None)


if __name__ == "__main__":
    main()
