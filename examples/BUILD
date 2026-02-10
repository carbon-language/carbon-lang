# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_binaries//:defs.bzl", "bazel_binaries")
load(
    "@rules_bazel_integration_test//bazel_integration_test:defs.bzl",
    "bazel_integration_tests",
    "integration_test_utils",
)
load("@rules_python//python:defs.bzl", "py_binary")
load("//bazel/carbon_rules:defs.bzl", "carbon_binary")

carbon_binary(
    name = "sieve",
    srcs = ["sieve.carbon"],
)

carbon_binary(
    name = "hello_world",
    srcs = ["hello_world.carbon"],
)

py_binary(
    name = "bazel_test_runner",
    testonly = True,
    srcs = ["bazel_test_runner.py"],
    data = ["//toolchain/install:install_data"],
    deps = [
        "@bazel_tools//tools/python/runfiles",
        "@rules_bazel_integration_test//bazel_integration_test/py:test_base",
    ],
)

bazel_example_files = [
    "bazel/BUILD",
    "bazel/MODULE.bazel",
    "bazel/example_lib.h",
    "bazel/example_lib.cpp",
    "bazel/example.cpp",
]

bazel_integration_tests(
    name = "bazel_min_test",
    bazel_binaries = bazel_binaries,
    bazel_versions = bazel_binaries.versions.all,
    # Override the default tags which assume a much more expensive test.
    tags = [],
    test_runner = ":bazel_test_runner",
    workspace_files = bazel_example_files,
    workspace_path = "bazel",
)

bazel_integration_tests(
    name = "bazel_full_test",
    bazel_binaries = bazel_binaries,
    bazel_versions = bazel_binaries.versions.all,
    env = {"CARBON_BAZEL_TEST_FULL": ""},
    # Override the default tags -- while running these manually is good as the
    # full tests are very expensive, there isn't a reason to run them
    # _exclusively_ as well.
    tags = ["manual"],
    test_runner = ":bazel_test_runner",
    workspace_files = bazel_example_files,
    workspace_path = "bazel",
)

# Convenience test suites with a simple names to run all the different
# configured Bazel versions of tests.
test_suite(
    name = "bazel_min_tests",
    tests = integration_test_utils.bazel_integration_test_names(
        "bazel_min_test",
        bazel_binaries.versions.all,
    ),
)

test_suite(
    name = "bazel_full_tests",
    # Full tests are too expensive to run by default.
    tags = ["manual"],
    tests = integration_test_utils.bazel_integration_test_names(
        "bazel_full_test",
        bazel_binaries.versions.all,
    ),
)
