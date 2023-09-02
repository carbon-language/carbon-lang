#!/usr/bin/env python3

"""Script to compute statistics about source code."""

from __future__ import annotations

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
from alive_progress import alive_bar  # type:ignore
from multiprocessing import Pool
import re
import termplotlib as tpl  # type:ignore
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from collections import Counter

# Test commit with a no-op comment.

BLANK_RE = re.compile(r"\s*")
COMMENT_RE = re.compile(r"\s*///*\s*")
LINE_RE = re.compile(
    r"""
    (?P<class_intro>\b(class|struct)\s+(?P<class_name>\w+)\b)|
    (?P<end_open_curly>{\s*(?P<open_curly_trailing_comment>//.*)?)|
    (?P<trailing_comment>//.*)|
    (?P<id>\b\w+\b)
""",
    re.X,
)


@dataclass
class Stats:
    """Stats collected while scanning source files"""

    lines: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    empty_comment_lines: int = 0
    comment_line_widths: Counter[int] = field(default_factory=lambda: Counter())
    lines_with_trailing_comments: int = 0
    classes: int = 0
    identifiers: int = 0
    identifier_widths: Counter[int] = field(default_factory=lambda: Counter())
    ids_per_line: Counter[int] = field(default_factory=lambda: Counter())

    def accumulate(self, other: Stats) -> None:
        self.lines += other.lines
        self.blank_lines += other.blank_lines
        self.empty_comment_lines += other.empty_comment_lines
        self.comment_lines += other.comment_lines
        self.comment_line_widths.update(other.comment_line_widths)
        self.lines_with_trailing_comments += other.lines_with_trailing_comments
        self.classes += other.classes
        self.identifiers += other.identifiers
        self.identifier_widths.update(other.identifier_widths)
        self.ids_per_line.update(other.ids_per_line)


def scan_file(file: Path) -> Stats:
    """Scans the provided file and accumulates stats."""
    stats = Stats()
    for line in file.open():
        # Strip off the line endings.
        line = line.rstrip("\r\n")
        # Skip over super long lines that are often URLs or structured data that
        # doesn't match "normal" source code patterns.
        if len(line) > 80:
            continue
        stats.lines += 1
        if re.fullmatch(BLANK_RE, line):
            stats.blank_lines += 1
            continue
        if m := re.match(COMMENT_RE, line):
            stats.comment_lines += 1
            if m.end() == len(line):
                stats.empty_comment_lines += 1
            else:
                stats.comment_line_widths[len(line)] += 1
            continue
        line_identifiers = 0
        for m in re.finditer(LINE_RE, line):
            if m.group("trailing_comment"):
                stats.lines_with_trailing_comments += 1
                break
            if m.group("class_intro"):
                stats.classes += 1
                line_identifiers += 1
                stats.identifier_widths[len(m.group("class_name"))] += 1
            elif m.group("end_open_curly"):
                pass
            else:
                assert m.group("id"), "Line is '%s', and match is '%s'" % (
                    line,
                    line[m.start() : m.end()],
                )
                line_identifiers += 1
                stats.identifier_widths[len(m.group("id"))] += 1
        stats.identifiers += line_identifiers
        stats.ids_per_line[line_identifiers] += 1
    return stats


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parsers command-line arguments and flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        metavar="FILE",
        type=Path,
        nargs="+",
        help="A file to scan while collecting statistics.",
    )
    return parser.parse_args(args=args)


def main() -> None:
    parsed_args = parse_args()
    stats = Stats()
    with alive_bar(len(parsed_args.files)) as bar:
        with Pool() as p:
            for file_stats in p.imap_unordered(scan_file, parsed_args.files):
                stats.accumulate(file_stats)
                bar()

    print(
        """
## Stats ##
Lines: %(lines)d
Blank lines: %(blank_lines)d
Comment lines: %(comment_lines)d
Empty comment lines: %(empty_comment_lines)d
Lines with trailing comments: %(lines_with_trailing_comments)d
Classes: %(classes)d
IDs: %(identifiers)d"""
        % asdict(stats)
    )

    def print_histogram(
        title: str, data: Dict[int, int], column_format: str
    ) -> None:
        print()
        print(title)
        key_min = min(data.keys())
        key_max = max(data.keys()) + 1
        values = [data.get(k, 0) for k in range(key_min, key_max)]
        keys = [column_format % k for k in range(key_min, key_max)]
        fig = tpl.figure()
        fig.barh(values, keys)
        fig.show()

    print_histogram(
        "## Comment line widths ##", stats.comment_line_widths, "%d columns"
    )
    print_histogram("## ID widths ##", stats.identifier_widths, "%d characters")
    print_histogram("## IDs per line ##", stats.ids_per_line, "%d ids")


if __name__ == "__main__":
    main()
