# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for building fuzz tests."""

def _append_if_not_present(kwargs, key, val):
    """Adds val to a list in kwargs indicated by key."""
    orig_list = kwargs.setdefault(key, [])
    if val not in orig_list:
        kwargs[key] = orig_list + [val]

def cc_fuzz_test(
        name,
        corpus,
        **kwargs):
    """Macro for C++ fuzzing test.

    In order to test the entire corpus, run:

        bazel test :fuzzer

    In order to test a single file, run:

        bazel run :fuzzer.bin -- $PWD/path/to/file

    Args:
        name: The main fuzz test rule name.
        corpus: List of files to use as a fuzzing corpus.
        **kwargs: Remaining arguments passed down to generated rules (with some
            kind-specific removals).
    """

    # Remove arguments which should only apply to the test.
    bin_kwargs = dict(kwargs)
    bin_kwargs.pop("size", None)

    # The fuzzer feature is required for fuzzer binaries.
    _append_if_not_present(bin_kwargs, "features", "fuzzer")

    # Provide an arg-less binary that can be run for testing specific files.
    bin_target = name + ".bin"
    native.cc_binary(
        name = bin_target,
        **bin_kwargs
    )

    # Remove arguments which should only apply to the binary.
    test_kwargs = dict(kwargs)
    for k in ("args", "data", "deps", "features", "srcs"):
        test_kwargs.pop(k, None)

    # Tag as a fuzz_test for convenience.
    _append_if_not_present(test_kwargs, "tags", "fuzz_test")

    # Append the corpus files to the test arguments. When run on a list of
    # files rather than a directory, libFuzzer-based fuzzers will perform a
    # regression test against the corpus.
    test_args = ["$(location %s)" % bin_target] + [
        "$(location %s)" % file
        for file in corpus
    ]

    # A regression test for the fuzzer corpus.
    native.sh_test(
        name = name,
        srcs = ["//bazel/fuzzing:fuzz_test.sh"],
        args = test_args,
        data = [bin_target] + corpus,
        **test_kwargs
    )
