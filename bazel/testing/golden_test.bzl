# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule for a golden test."""

def golden_test(name, golden, cmd, data, golden_is_subset = False, **kwargs):
    """Compares two files. Passes if they are identical.

    Args:
      name: Name of the build rule.
      cmd: The command whose output is being tested.
      golden: The golden file to be compared against the command output.
      data: Data files.
      golden_is_subset: Set to True if the golden file should be a subset of
        command output.
      **kwargs: Any additional parameters for the generated py_test.
    """
    args = ["$(location %s)" % golden, cmd]
    if golden_is_subset:
        args.append("--golden_is_subset")
    native.py_test(
        name = name,
        srcs = ["//bazel/testing:golden_test.py"],
        main = "//bazel/testing:golden_test.py",
        args = args,
        data = [golden] + data,
        env = {
            # TODO(#580): Remove this when leaks are fixed.
            "ASAN_OPTIONS": "detect_leaks=0",
        },
        **kwargs
    )
