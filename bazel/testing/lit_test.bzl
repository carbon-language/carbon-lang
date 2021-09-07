# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule for a lit test."""

def lit_test(name, test_path, data, args = None, env = None, **kwargs):
    """Compares two files. Passes if they are identical.

    Args:
      name: Name of the build rule.
      data: Data files.
      env: Optional environment.
      **kwargs: Any additional parameters for the generated py_test.
    """
    if not args:
        args = []
    if not env:
        env = {}
    native.py_test(
        name = name,
        srcs = ["//bazel/testing:lit_test.py"],
        main = "//bazel/testing:lit_test.py",
        data = ["@llvm-project//llvm:lit"] + data,
        args = [test_path] + args,
        env = env,
        **kwargs
    )
