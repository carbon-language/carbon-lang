#!/usr/bin/env python3

"""Checks diagnostic use.

Validates that each diagnostic declared with CARBON_DIAGNOSTIC_KIND is
referenced by one (and only one) CARBON_DIAGNOSTIC.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import collections
from concurrent import futures
import itertools
from pathlib import Path
import os
import re
import sys
from typing import Dict, List, NamedTuple, Set


# Example or test diagnostics, ignored because they're expected to not pass.
IGNORED = set(["MyDiagnostic", "TestDiagnostic", "TestDiagnosticNote"])


class Location(NamedTuple):
    """A location for a diagnostic."""

    def __str__(self) -> str:
        return f"{str(self.path)}:{self.line}"

    path: Path
    line: int


def load_diagnostic_kind() -> Set[str]:
    """Returns the set of declared diagnostic kinds.

    This isn't validated for uniqueness because the compiler does that.
    """
    path = Path("toolchain/diagnostics/diagnostic_registry.def")
    content = path.read_text()
    decls = set(re.findall(r"CARBON_DIAGNOSTIC_KIND\((.+)\)", content))
    return decls.difference(IGNORED)


def load_diagnostic_uses_in(
    path: Path,
) -> Dict[str, List[Location]]:
    """Returns the path's CARBON_DIAGNOSTIC uses."""
    content = path.read_text()

    # Keep a line cursor so that we don't keep re-scanning the file.
    line = 1
    line_offset = 0

    found: Dict[str, List[Location]] = collections.defaultdict(lambda: [])
    for m in re.finditer(r"CARBON_DIAGNOSTIC\(\s*(\w+),", content):
        diag = m.group(1)
        if diag in IGNORED:
            continue
        line += content.count("\n", line_offset, m.start())
        line_offset = m.start()
        found[diag].append(Location(path, line))
    return found


def load_diagnostic_uses() -> Dict[str, List[Location]]:
    """Returns all CARBON_DIAGNOSTIC uses."""
    globs = itertools.chain(
        *[Path("toolchain").glob(f"**/*.{ext}") for ext in ("h", "cpp")]
    )
    with futures.ThreadPoolExecutor() as exec:
        results = exec.map(load_diagnostic_uses_in, globs)

    found: Dict[str, List[Location]] = collections.defaultdict(lambda: [])
    for result in results:
        for diag, locations in result.items():
            found[diag].extend(locations)
    return found


def check_uniqueness(uses: Dict[str, List[Location]]) -> bool:
    """If any diagnostic is non-unique, prints an error and returns true."""
    has_errors = False
    for diag in sorted(uses.keys()):
        if len(uses[diag]) > 1:
            print(f"Non-unique diagnostic {diag}:", file=sys.stderr)
            for loc in uses[diag]:
                print(f"  - {loc}", file=sys.stderr)
            has_errors = True
    return has_errors


def check_unused(decls: Set[str], uses: Dict[str, List[Location]]) -> bool:
    """If any diagnostic is unused, prints an error and returns true."""
    unused = decls.difference(uses.keys())
    for diag in sorted(unused):
        print(f"Unused diagnostic: {diag}")
    return False


def main() -> None:
    # Run from the repo root.
    os.chdir(Path(__file__).parent.parent.parent)
    decls = load_diagnostic_kind()
    uses = load_diagnostic_uses()

    if any([check_uniqueness(uses), check_unused(decls, uses)]):
        exit(1)


if __name__ == "__main__":
    main()
