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

from pathlib import Path
import os
import re
from typing import Set


def load_diagnostic_kind() -> Set[str]:
    """Returns the set of declared diagnostic kinds.

    This isn't validated for uniqueness because the compiler does that.
    """
    path = Path("toolchain/diagnostics/diagnostic_registry.def")
    content = path.read_text()
    return set(re.findall(r"CARBON_DIAGNOSTIC_KIND\((.+)\)", content))


def check_diagnostic_uses(diagnostics: Set[str]) -> bool:
    """Matches diagnostic uses to kinds.

    Returns True if an error was found.
    """
    ignored = set(["DiagnosticName", "MyDiagnostic", "TestDiagnostic"])
    unused_diagnostics = diagnostics.difference(ignored)
    has_errors = False

    for ext in ("h", "cpp"):
        for path in Path("toolchain").glob(f"**/*.{ext}"):
            content = path.read_text()
            defined_diagnostics = re.findall(
                r"CARBON_DIAGNOSTIC\((?:\s|\n)*(\w+),", content
            )
            for d in defined_diagnostics:
                if d in unused_diagnostics:
                    unused_diagnostics.remove(d)
                elif d in ignored:
                    continue
                else:
                    has_errors = True
                    if d in diagnostics:
                        print(f"Diagnostic used twice: {d}")
                    else:
                        # Should be a compile error, but handle it to make sure
                        # we're not missing things.
                        print(f"Diagnostic not declared: {d}")

    if unused_diagnostics:
        has_errors = True
        unused_diagnostics_str = ", ".join(sorted(list(unused_diagnostics)))
        print(f"Diagnostics not used: {unused_diagnostics_str}")

    return has_errors


def main() -> None:
    # Run from the repo root.
    os.chdir(Path(__file__).parent.parent.parent)
    diagnostics = load_diagnostic_kind()
    if check_diagnostic_uses(diagnostics):
        exit(1)


if __name__ == "__main__":
    main()
