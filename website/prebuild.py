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
    title = re.sub("`([^`]+)`", r"<code>\1</code>", title)
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
    subdir_str: str,
    top_nav_order: int,
    parent_title: Optional[str] = None,
    grandchild_dirs: bool = False,
) -> None:
    """Automatically adds child information to a subdirectory's markdown files.

    This is in support of navigation.
    """
    assert not (parent_title and grandchild_dirs)

    subdir = Path(subdir_str)
    readme = subdir / "README.md"
    readme_content = readme.read_text()

    def include_child(md_file: Path) -> bool:
        """Returns whether md_file is a child of this label."""
        # The top-level README.md isn't a child.
        if md_file == readme:
            return False
        # Skip directories that aren't part of the repository.
        if "/node_modules/" in str(md_file):
            return False
        if "/dist/" in str(md_file):
            return False
        return True

    readme_title = get_title(readme, readme_content)
    readme_titles = [readme_title]
    if parent_title:
        readme_titles.insert(0, parent_title)
    children = [x for x in subdir.glob("**/*.md") if include_child(x)]
    add_frontmatter(
        readme, readme_content, readme_titles, top_nav_order, bool(children)
    )

    if grandchild_dirs:
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
            # Skip files that aren't proposals.
            if not m:
                continue
            child_title = f"#{m[1]}: {child_title}"
            child_nav_order = int(m[1])

        titles = [readme_title, child_title]
        has_children = False
        if parent_title:
            titles.insert(0, parent_title)
        elif grandchild_dirs and child.parent in child_dirs:
            if child.name == "README.md":
                has_children = child_dirs[child.parent].has_grandchildren
            else:
                dir_title = child_dirs[child.parent].title
                titles.insert(1, dir_title)

        add_frontmatter(
            child, child_content, titles, child_nav_order, has_children
        )


def label_root_file(
    name: str, title: str, top_nav_order: int, has_children: bool = False
) -> None:
    """Adds frontmatter to a root file, like CONTRIBUTING.md."""
    f = Path(name)
    add_frontmatter(f, f.read_text(), [title], top_nav_order, has_children)


def main() -> None:
    # Ensure this runs from the repo root.
    os.chdir(Path(__file__).parents[1])

    # The external symlink is created by scripts/create_compdb.py, and can
    # interfere with local execution.
    external = Path("external")
    if external.exists():
        external.unlink()

    # Move files to the repo root.
    for f in Path("website").iterdir():
        if f.name in ["README.md", ".jekyll-cache", "_site"]:
            continue
        f.rename(f.name)

    # Use an object for a reference.
    nav_order = [0]

    # Returns an incrementing value for ordering.
    def next(nav_order: list[int]) -> int:
        nav_order[0] += 1
        return nav_order[0]

    label_root_file("README.md", "Home", next(nav_order))
    label_root_file("CONTRIBUTING.md", "Contributing", next(nav_order))
    label_subdir("docs/design", next(nav_order), grandchild_dirs=True)
    label_subdir("docs/guides", next(nav_order))
    label_subdir("docs/project", next(nav_order), grandchild_dirs=True)
    label_subdir("docs/spec", next(nav_order))
    # Provide a small file to cluster implementation-related directories.
    label_root_file(
        "implementation.md",
        "Implementation",
        next(nav_order),
        has_children=True,
    )
    label_subdir("utils", next(nav_order))
    label_subdir("proposals", next(nav_order))
    label_root_file("CODE_OF_CONDUCT.md", "Code of conduct", next(nav_order))
    label_root_file("SECURITY.md", "Security policy", next(nav_order))

    # Reset the order for the implementation children.
    nav_order[0] = 0
    label_subdir(
        "toolchain/docs", next(nav_order), parent_title="Implementation"
    )
    label_subdir("explorer", next(nav_order), parent_title="Implementation")
    label_subdir("testing", next(nav_order), parent_title="Implementation")


if __name__ == "__main__":
    main()
