# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule for a lit test."""

def glob_lit_tests(driver, data, test_file_exts, **kwargs):
    """Runs `lit` on test_dir.

    `lit` reference:
      https://llvm.org/docs/CommandGuide/lit.html

    Args:
      driver: The path to the lit config.
      data: A list of tools to provide to the tests. These will be aliased for
        execution.
      test_file_exts: A list of extensions to use for tests.
      **kwargs: Any additional parameters for the generated py_test.
    """
    test_files = native.glob(
        ["**"],
        exclude_directories = 1,
    )
    data.append("@llvm-project//llvm:lit")
    for f in test_files:
        if f.split(".")[-1] not in test_file_exts:
            continue
        native.py_test(
            name = "%s.test" % f,
            srcs = ["//bazel/testing:lit_test.py"],
            main = "//bazel/testing:lit_test.py",
            data = data + [driver, f],
            args = ["--package_name=%s" % native.package_name(), "--"],
            **kwargs
        )
