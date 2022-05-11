#!/usr/bin/env python3

"""Requires files be named for their SHA1.

We name fuzzer corpus files for their SHA1. The choice of SHA1 is for
consistency with git.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import hashlib
from pathlib import Path
import sys


def main() -> None:
    has_errors = False
    bad_files = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        with path.open("rb") as f:
            want = hashlib.sha1(f.read()).hexdigest()
        if path.name != want:
            want_path = path.parent.joinpath(want)
            bad_files.append((path, want_path))
            print(f"mv {path} {want_path}", file=sys.stderr)
            has_errors = True
    if has_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
