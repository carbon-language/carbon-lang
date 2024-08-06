#!/usr/bin/env python3

"""Updates files in preparation for a jekyll build.

Used from .github/workflows/gh_pages.yaml. This updates the file and directory
structure prior to the jekyll build.
"""

import dataclasses
import os
from pathlib import Path
import re
from typing import Optional

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""


@dataclasses.dataclass
class ChildDir:
    """Tracks whether a child directory has grandchildren."""

    title: str
    has_grandchildren: bool = False


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
    titles: list[str],
    nav_order: Optional[int],
    has_children: bool,
) -> None:
    """Adds frontmatter to a file."""
    content = "---\n"

    assert len(titles) <= 3
    content += f'title: "{titles[-1]}"\n'
    if len(titles) >= 2:
        content += f'parent: "{titles[-2]}"\n'
    if len(titles) == 3:
        content += f'grand_parent: "{titles[-3]}"\n'

    if nav_order is not None:
        content += f"nav_order: {nav_order}\n"
    if has_children:
        # has_toc is disabling the list of child files in a separate table of
        # contents.
        content += "has_children: true\nhas_toc: false\n"
    content += "---\n\n" + orig_content
    f.write_text(content)


def label_subdir(
    subdir_str: str, top_nav_order: int, use_grandchildren: bool = False
) -> None:
    """Automatically adds child information to a subdirectory's markdown files.

    This is in support of navigation.
    """
    subdir = Path(subdir_str)
    readme = subdir / "README.md"
    readme_content = readme.read_text()
    readme_title = get_title(readme, readme_content)

    children = [x for x in subdir.glob("**/*.md") if x != readme]
    add_frontmatter(
        readme, readme_content, [readme_title], top_nav_order, bool(children)
    )

    if use_grandchildren:
        # When adding grandchildren, we cluster files by child directory.
        child_dirs = {}
        for child in children:
            if child.name == "README.md":
                title = get_title(child, child.read_text())
                child_dirs[child.parent] = ChildDir(title)
        for child in children:
            if child.name != "README.md" and child.parent in child_dirs:
                child_dirs[child.parent].has_grandchildren = True

    for child in children:
        child_content = child.read_text()
        child_title = get_title(child, child_content)
        child_nav_order = None

        if subdir_str == "proposals":
            # Use proposal numbers as part of the title and ordering.
            m = re.match(r"p(\d+).md", child.name)
            if m:
                child_title = f"#{m[1]}: {child_title}"
                child_nav_order = int(m[1])

        titles = [readme_title, child_title]
        has_children = False
        if use_grandchildren and child.parent in child_dirs:
            if child.name == "README.md":
                has_children = child_dirs[child.parent].has_grandchildren
            else:
                parent_title = child_dirs[child.parent].title
                titles = [readme_title, parent_title, child_title]

        add_frontmatter(
            child, child_content, titles, child_nav_order, has_children
        )


def label_root_file(name: str, title: str, top_nav_order: int) -> None:
    """Adds frontmatter to a root file, like CONTRIBUTING.md."""
    f = Path(name)
    add_frontmatter(f, f.read_text(), [title], top_nav_order, False)


def main() -> None:
    # Ensure this runs from the repo root.
    os.chdir(Path(__file__).parent.parent)

    # bazel-execroot interferes with jekyll because it's a broken symlink.
    Path("bazel-execroot").unlink()

    # Move files to the repo root.
    for f in Path("website").iterdir():
        f.rename(f.name)

    # Use an object for a reference.
    nav_order = [0]

    # Returns an incrementing value for ordering.
    def next(nav_order: list[int]) -> int:
        nav_order[0] += 1
        return nav_order[0]

    label_root_file("README.md", "README", next(nav_order))
    label_root_file("CONTRIBUTING.md", "Contributing", next(nav_order))
    label_subdir("docs/design", next(nav_order), True)
    label_subdir("docs/guides", next(nav_order))
    label_subdir("docs/project", next(nav_order), True)
    label_subdir("docs/spec", next(nav_order))
    label_subdir("toolchain", next(nav_order))
    label_subdir("explorer", next(nav_order))
    label_subdir("testing", next(nav_order))
    label_subdir("utils", next(nav_order))
    label_subdir("proposals", next(nav_order))
    label_root_file("CODE_OF_CONDUCT.md", "Code of conduct", next(nav_order))
    label_root_file("SECURITY.md", "Security policy", next(nav_order))


if __name__ == "__main__":
    main()
