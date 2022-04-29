#!/usr/bin/env python3

"""Update the roots of the Carbon build used for dependency checking.

The dependency checking cannot use wildcard queries, so we use them here and
then create lists of relevant roots in the build file.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import subprocess
from pathlib import Path

# Change the working directory to the repository root so that the remaining
# operations reliably operate relative to that root.
os.chdir(Path(__file__).parent.parent.parent)

print("Compute non-test C++ root targets...")
non_test_cc_roots_query = subprocess.check_output(
    [
        "./scripts/run_bazel.py",
        "query",
        "--noshow_progress",
        "--noimplicit_deps",
        "--notool_deps",
        "--output=minrank",
        (
            'let non_tests = kind("cc.* rule", attr(testonly, 0, //...))'
            ' in kind("cc.* rule", deps($non_tests))'
        ),
    ],
    universal_newlines=True,
)
ranked_targets = [line.split() for line in non_test_cc_roots_query.splitlines()]
roots = [target for rank, target in ranked_targets if int(rank) == 0]
print("Found roots:\n%s" % "\n".join(roots))

print("Replace non-test C++ roots in the BUILD file...")
buildozer_run = subprocess.run(
    [
        "./scripts/run_buildozer.py",
        "remove data",
    ]
    + ["add data '%s'" % root for root in roots]
    + ["//bazel/check_deps:non_test_cc_rules"],
)
if buildozer_run.returncode == 3:
    print("No changes needed!")
else:
    buildozer_run.check_returncode()
    print("Successfully updated roots in the BUILD file!")
