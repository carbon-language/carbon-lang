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

BLANK_RE = re.compile(r"\s*")
COMMENT_RE = re.compile(r"\s*///*\s*")
LINE_RE = re.compile(
    r"""
    (?P<class_intro>\b(class|struct)\s+(?P<class_name>\w+)\b)|
    (?P<end_open_curly>{\s*(?P<open_curly_trailing_comment>//.*)?)|
    (?P<trailing_comment>//.*)|
    (?P<internal_comment>/\*.*\*/)|
    (?P<string_literal>"([^"]|\\")*"|'([^']|\\')*')|
    (?P<float_literal>\b(0[xb][0-9a-fA-F']*|[0-9][0-9']*)\.[0-9a-fA-F']*([eEpP][0-9a-fA-F']*)?)|
    (?P<int_literal>\b(0[xb][0-9a-fA-F']+|[0-9][0-9']*)([eEpP][0-9a-fA-F']*)?)|
    (?P<symbol>[\[\]{}(),.;]|[-+=!@#$%^&*/?|<>]+)|
    (?P<keyword>\b(auto|bool|break|case|catch|char|class|const|continue|default|do|double|else|enum|explicit|extern|false|float|for|friend|goto|if|inline|int|long|mutable|namespace|new|nullptr|operator|private|protected|public|return|short|signed|sizeof|static|struct|switch|template|this|throw|true|try|typedef|union|unsigned|using|virtual|void|while)\b)|
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
    internal_comments: int = 0
    string_literals: int = 0
    string_literals_per_line: Counter[int] = field(
        default_factory=lambda: Counter()
    )
    int_literals: int = 0
    int_literals_per_line: Counter[int] = field(
        default_factory=lambda: Counter()
    )
    float_literals: int = 0
    float_literals_per_line: Counter[int] = field(
        default_factory=lambda: Counter()
    )
    symbols: int = 0
    symbols_per_line: Counter[int] = field(default_factory=lambda: Counter())
    keywords: int = 0
    keywords_per_line: Counter[int] = field(default_factory=lambda: Counter())
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
        self.internal_comments += other.internal_comments
        self.string_literals += other.string_literals
        self.string_literals_per_line.update(other.string_literals_per_line)
        self.int_literals += other.int_literals
        self.int_literals_per_line.update(other.int_literals_per_line)
        self.float_literals += other.float_literals
        self.float_literals_per_line.update(other.float_literals_per_line)
        self.symbols += other.symbols
        self.symbols_per_line.update(other.symbols_per_line)
        self.keywords += other.keywords
        self.keywords_per_line.update(other.keywords_per_line)
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
        line_string_literals = 0
        line_int_literals = 0
        line_float_literals = 0
        line_symbols = 0
        line_keywords = 0
        line_identifiers = 0
        for m in re.finditer(LINE_RE, line):
            if m.group("trailing_comment"):
                stats.lines_with_trailing_comments += 1
                break
            if m.group("class_intro"):
                stats.classes += 1
                line_keywords += 1
                line_identifiers += 1
                stats.identifier_widths[len(m.group("class_name"))] += 1
            elif m.group("end_open_curly"):
                line_symbols += 1
            elif m.group("internal_comment"):
                stats.internal_comments += 1
            elif m.group("string_literal"):
                line_string_literals += 1
            elif m.group("int_literal"):
                line_int_literals += 1
            elif m.group("float_literal"):
                line_float_literals += 1
            elif m.group("symbol"):
                line_symbols += 1
            elif m.group("keyword"):
                line_keywords += 1
            else:
                assert m.group("id"), "Line is '%s', and match is '%s'" % (
                    line,
                    line[m.start() : m.end()],
                )
                line_identifiers += 1
                stats.identifier_widths[len(m.group("id"))] += 1
        stats.string_literals += line_string_literals
        stats.string_literals_per_line[line_string_literals] += 1
        stats.int_literals += line_int_literals
        stats.int_literals_per_line[line_int_literals] += 1
        stats.float_literals += line_float_literals
        stats.float_literals_per_line[line_float_literals] += 1
        stats.symbols += line_symbols
        stats.symbols_per_line[line_symbols] += 1
        stats.keywords += line_keywords
        stats.keywords_per_line[line_keywords] += 1
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
Internal comments: %(internal_comments)d
String literals: %(string_literals)d
Int literals: %(int_literals)d
Float literals: %(float_literals)d
Symbols: %(symbols)d
Keywords: %(keywords)d
IDs: %(identifiers)d"""
        % asdict(stats)
    )

    tokens = (
        stats.string_literals
        + stats.int_literals
        + stats.float_literals
        + stats.symbols
        + stats.keywords
        + stats.identifiers
    )
    print(
        f"""
Fraction of blank lines: {stats.blank_lines / stats.lines}
Fraction of comment lines: {stats.comment_lines / stats.lines}

Total counted tokens: {tokens}
Fraction string literals: {stats.string_literals / tokens}
Fraction int literals: {stats.int_literals / tokens}
Fraction float literals: {stats.float_literals / tokens}
Fraction symbols: {stats.symbols / tokens}
Fraction keywords: {stats.keywords / tokens}
Fraction IDs: {stats.identifiers / tokens}
    """
    )

    def print_histogram(
        title: str, data: Dict[int, int], column_format: str
    ) -> None:
        print()
        key_min = min(data.keys())
        key_max = max(data.keys()) + 1
        values = [data.get(k, 0) for k in range(key_min, key_max)]
        keys = [column_format % k for k in range(key_min, key_max)]
        total = sum(values)
        median = key_min
        count = total
        for k in range(key_min, key_max):
            count -= data.get(k, 0)
            if count <= total / 2:
                median = k
                break

        print(title + f" (median: {median})")
        fig = tpl.figure()
        fig.barh(values, keys)
        fig.show()

    print_histogram(
        "## Comment line widths ##", stats.comment_line_widths, "%d columns"
    )

    print_histogram(
        "## String literals per line ##",
        stats.string_literals_per_line,
        "%d literals",
    )
    print_histogram(
        "## Int literals per line ##",
        stats.int_literals_per_line,
        "%d literals",
    )
    print_histogram(
        "## Float literals per line ##",
        stats.float_literals_per_line,
        "%d literals",
    )
    print_histogram(
        "## Symbols per line ##", stats.symbols_per_line, "%d symbols"
    )
    print_histogram(
        "## Keywords per line ##", stats.keywords_per_line, "%d keywords"
    )

    print_histogram("## ID widths ##", stats.identifier_widths, "%d characters")
    print_histogram("## IDs per line ##", stats.ids_per_line, "%d ids")


if __name__ == "__main__":
    main()
