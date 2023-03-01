# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for building fuzz tests."""

load("@rules_cc//cc:defs.bzl", "cc_test")

def _cc_fuzz_test(corpus, args, data, **kwargs):
    """Generates a single test target.

    Append the corpus files to the test arguments. When run on a list of
    files rather than a directory, libFuzzer-based fuzzers will perform a
    regression test against the corpus.
    """
    cc_test(
        data = data + corpus,
        args = args + ["$(location %s)" % file for file in corpus],
        **kwargs
    )

def cc_fuzz_test(
        name,
        corpus,
        args = [],
        data = [],
        features = [],
        tags = [],
        deps = [],
        shard_count = 1,
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
        shard_count: Provides sharding of the fuzz test.
        **kwargs: Remaining arguments passed down to the fuzz test.
    """

    # Add relevant tag and feature if necessary.
    if "fuzz_test" not in tags:
        tags = tags + ["fuzz_test"]
    if "fuzzer" not in features:
        features = features + ["fuzzer"]
    if "@llvm-project//compiler-rt:FuzzerMain" not in deps:
        deps = deps + ["@llvm-project//compiler-rt:FuzzerMain"]

    # The FuzzerMain library doesn't support sharding based on inputs, so we
    # general separate test targets in order to shard execution.
    if shard_count == 1:
        # When there's one shard, only one target is needed.
        _cc_fuzz_test(
            corpus,
            args,
            data,
            name = name,
            features = features,
            tags = tags,
            deps = deps,
            **kwargs
        )
    else:
        # Calculate the number of inputs per shard. This is equivalent to
        # ceiling division, so that the corpus subsetting doesn't miss odd
        # files.
        shard_size = len(corpus) // shard_count
        if shard_count * shard_size < len(corpus):
            shard_size += 1

        # Create separate targets for each shard.
        shards = []
        for shard in range(shard_count):
            shard_name = "{0}.shard{1}".format(name, shard)
            shards.append(shard_name)

            _cc_fuzz_test(
                corpus[shard * shard_size:(shard + 1) * shard_size],
                args,
                data,
                name = shard_name,
                features = features,
                tags = tags,
                deps = deps,
                **kwargs
            )

        # Create a suite containing all shards.
        native.test_suite(
            name = name,
            tests = shards,
        )

        # Create one target that includes the full corpus.
        _cc_fuzz_test(
            corpus,
            args,
            data,
            name = "{0}.full_corpus".format(name),
            features = features,
            tags = tags + ["manual"],
            deps = deps,
            **kwargs
        )
