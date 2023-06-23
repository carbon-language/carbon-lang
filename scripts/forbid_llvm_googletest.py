#!/usr/bin/env python3

"""Detects and prevents dependencies on LLVM's googletest.

Carbon uses googletest directly, and it's a significantly more recent version
than is provided by LLVM. Using both versions in the same binary leads to
problems, so this detects dependencies.

We also have some dependency checking at //bazel/check_deps. This is a separate
script because check_deps relies on being able to validate specific binaries
which change infrequently, whereas this effectively monitors all cc_test rules,
the set of which is expected to be altered more often.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import subprocess

import scripts_utils

_MESSAGE = """\
Dependencies on @llvm-project//llvm:gtest are forbidden, but a dependency path
was detected:

%s
Carbon uses GoogleTest through @com_google_googletest, which is a different
version than LLVM uses at @llvm-project//llvm:gtest. As a consequence,
dependencies on @llvm-project//llvm:gtest must be avoided.
"""


def main() -> None:
    scripts_utils.chdir_repo_root()
    args = [
        scripts_utils.locate_bazel(),
        "query",
        "somepath(//..., @llvm-project//third-party/unittest:gtest)",
    ]
    p = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
    )
    if p.returncode != 0:
        print(p.stderr)
        exit(f"bazel query returned {p.returncode}")
    if p.stdout:
        exit(_MESSAGE % p.stdout)
    print("Done!")


if __name__ == "__main__":
    main()
