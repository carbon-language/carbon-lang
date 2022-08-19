# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for building fuzz tests."""

load("@rules_cc//cc:defs.bzl", "cc_test")

def cc_fuzz_test(
        name,
        corpus,
        args = [],
        data = [],
        features = [],
        tags = [],
        deps = [],
        **kwargs):
    """Macro for C++ fuzzing test.

    In order to run tests on a single file, run the fuzzer binary under
    bazel-bin directly. That will avoid the args being passed by Bazel.

    Args:
        name: The main fuzz test rule name.
        corpus: List of files to use as a fuzzing corpus.
        args: Will have the locations of the corpus files added and passed down
            to the fuzz test.
        data: Will have the corpus added and passed down to the fuzz test.
        features: Will have the "fuzzer" feature added and passed down to the
            fuzz test.
        tags: Will have "fuzz_test" added and passed down to the fuzz test.
        deps: Will have "@llvm-project//compiler-rt:FuzzerMain" added and passed
            down to the fuzz test.
        **kwargs: Remaining arguments passed down to the fuzz test.
    """

    # Add relevant tag and feature if necessary.
    if "fuzz_test" not in tags:
        tags = tags + ["fuzz_test"]
    if "fuzzer" not in features:
        features = features + ["fuzzer"]
    if "@llvm-project//compiler-rt:FuzzerMain" not in deps:
        deps = deps + ["@llvm-project//compiler-rt:FuzzerMain"]

    # Append the corpus files to the test arguments. When run on a list of
    # files rather than a directory, libFuzzer-based fuzzers will perform a
    # regression test against the corpus.
    data = data + corpus
    args = args + ["$(location %s)" % file for file in corpus]

    cc_test(
        name = name,
        args = args,
        data = data,
        features = features,
        tags = tags,
        deps = deps,
        **kwargs
    )
