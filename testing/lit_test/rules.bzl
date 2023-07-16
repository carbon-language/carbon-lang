# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule for a lit test."""

def glob_lit_tests(name, driver, data, test_file_exts, exclude = None, **kwargs):
    """Runs `lit` on test_dir.

    `lit` reference:
      https://llvm.org/docs/CommandGuide/lit.html

    This will generate three types of rules:
    1. For each test file, a rule with ".test" at the end. For example,
       "foo/bar/baz.carbon" will generate ":foo/bar/baz.carbon.test"
    2. If any test exists in subdirectory, a test_suite is generated for each
       directory. For example, ":foo" will run all tests under directory "foo",
       and ":foo/bar" will run all tests under "foo/bar".
    3. A test_suite containing all tests. The name of this will be the "name"
       arg passed to the glob_lit_tests rule.

    Args:
      name: The name of the test_suite rule to generate for running all tests.
      driver: The path to the lit config.
      data: A list of tools to provide to the tests. These will be aliased for
        execution.
      test_file_exts: A list of extensions to use for tests.
      exclude: A list of paths to exclude from the glob.
      **kwargs: Any additional parameters for the generated py_test.
    """
    if not exclude:
        exclude = []
    test_files = native.glob(
        ["**"],
        exclude = exclude,
        exclude_directories = 1,
    )
    data.append("@llvm-project//llvm:lit")
    suites = dict()
    all_tests = list()
    for f in test_files:
        if f.split(".")[-1] not in test_file_exts:
            continue
        test = "%s.test" % f
        all_tests.append(test)
        native.py_test(
            name = test,
            srcs = ["//testing/lit_test:lit_test.py"],
            main = "//testing/lit_test:lit_test.py",
            data = data + [driver, f],
            args = ["--package_name=%s" % native.package_name(), "--"],
            size = "small",
            **kwargs
        )

        # Cluster tests by directory in order to produce suites. For example,
        # foo/bar/baz.carbon.test is added to suites :foo and :foo/bar.
        dirs = f.split("/")[:-1]
        for num_parts in range(1, 1 + len(dirs)):
            dir = "/".join(dirs[:num_parts])
            if dir not in suites:
                suites[dir] = []
            suites[dir].append(test)
    native.test_suite(name = name, tests = all_tests)
    for suite, tests in suites.items():
        native.test_suite(name = suite, tests = tests)
