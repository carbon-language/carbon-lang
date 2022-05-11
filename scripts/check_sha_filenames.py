#!/usr/bin/env python3

"""Requires files be named for their SHA1.

We name fuzzer corpus files for their SHA1. The choice of SHA1 is for
consistency with git.

This maintains the current extension for .textproto, but at some point we might
want to specify the extension by path.
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
    for arg in sys.argv[1:]:
        path = Path(arg)
        with path.open("rb") as f:
            content = f.read()
        if len(content) == 0:
            want = "empty"
        else:
            want = hashlib.sha1(content).hexdigest()
        want_path = path.parent.joinpath(want).with_suffix(path.suffix)
        if path != want_path:
            print(f"Renaming {path} to {want_path}", file=sys.stderr)
            path.rename(want_path)
            has_errors = True
    if has_errors:
        exit(1)


if __name__ == "__main__":
    main()
