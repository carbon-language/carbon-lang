# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule for a golden ttest."""

def golden_test(name, golden, subject, **kwargs):
    """Compares two files. Passes if they are identical.

    Args:
      name: Name of the build rule.
      subject: The generated file to be compared.
      golden: The golden file to be compared.
      **kwargs: Any additional parameters for the generated sh_test.
    """
    native.sh_test(
        name = name,
        srcs = ["//bazel/testing:golden_test.sh"],
        args = [
            "$(location %s)" % golden,
            "$(location %s)" % subject,
        ],
        data = [golden, subject],
        **kwargs
    )
