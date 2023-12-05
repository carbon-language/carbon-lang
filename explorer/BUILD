# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("//bazel/cc_toolchains:defs.bzl", "cc_env")
load("//bazel/sh_run:rules.bzl", "glob_sh_run")
load("//testing/file_test:rules.bzl", "file_test")

package(default_visibility = [
    "//bazel/check_deps:__pkg__",
    "//explorer:__subpackages__",
    "//installers:__subpackages__",
])

filegroup(
    name = "standard_libraries",
    srcs = ["data/prelude.carbon"],
)

cc_library(
    name = "main",
    srcs = ["main.cpp"],
    hdrs = ["main.h"],
    deps = [
        "//common:error",
        "//common:init_llvm",
        "//common:ostream",
        "//explorer/base:trace_stream",
        "//explorer/parse_and_execute",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "explorer",
    srcs = ["main_bin.cpp"],
    env = cc_env(),
    deps = [
        ":main",
        "//common:bazel_working_dir",
        "@llvm-project//llvm:Support",
    ],
)

EXPLORER_GTEST_TESTS = [
    (
        "ast",
        "ast_test_matchers_test",
        "ASTTestMatchers",
    ),
    (
        "ast",
        "expression_test",
        "ExpressionTest",
    ),
    (
        "ast",
        "element_test",
        "ElementTest",
    ),
    (
        "ast",
        "pattern_test",
        "PatternTest",
    ),
    (
        "base",
        "arena_test",
        "ArenaTest",
    ),
    (
        "base",
        "decompose_test",
        "HashFromDecomposeTest",
    ),
    (
        "base",
        "error_builders_test",
        "ErrorBuildersTest",
    ),
    (
        "base",
        "set_file_context_raii_test",
        "SetFileContextRaiiTest",
    ),
    (
        "base",
        "set_program_phase_raii_test",
        "SetProgramPhaseRaiiTest",
    ),
    (
        "parse_and_execute",
        "parse_and_execute_test",
        "ParseAndExecuteTest",
    ),
    (
        "syntax",
        "parse_test",
        "ParseTest",
    ),
    (
        "syntax",
        "unimplemented_example_test",
        "UnimplementedExampleTest",
    ),
]

# All explorer tests are packaged into a single binary to reduce disk usage.
cc_binary(
    name = "explorer_tests",
    testonly = 1,
    srcs = ["file_test.cpp"],
    deps = [
        ":main",
        "//common:check",
        "//testing/base:test_raw_ostream",
        "//testing/file_test:file_test_base",
        "@com_google_absl//absl/flags:flag",
        "@com_googlesource_code_re2//:re2",
    ] + ["//explorer/%s:%s" % (test_dir, test_target) for (test_dir, test_target, test_suite) in EXPLORER_GTEST_TESTS],
)

file_test(
    name = "file_test",
    size = "small",
    args = ["--gtest_filter=ExplorerFileTest.*"],
    prebuilt_binary = ":explorer_tests",
    shard_count = 20,
    tests = glob(["testdata/**/*.carbon"]),
)

file_test(
    name = "file_test.trace",
    size = "small",
    args = [
        "--trace",
        "--gtest_filter=ExplorerFileTest.*",
    ],
    prebuilt_binary = ":explorer_tests",
    shard_count = 30,
    tests = glob(
        ["testdata/**/*.carbon"],
        exclude = [
            # `limits` tests check for various limit conditions (such as an
            # infinite loop). The tests collectively don't test tracing
            # because it creates substantial additional overhead.
            "testdata/limits/**",
            # `trace` tests do tracing by default.
            "testdata/trace/**",
            # Expensive tests to trace.
            "testdata/assoc_const/rewrite_large_type.carbon",
            "testdata/linked_list/typed_linked_list.carbon",
        ],
    ),
)

[
    sh_test(
        name = "%s_%s" % (test_dir, test_target),
        size = "small",
        srcs = ["explorer_tests"],
        args = [
            "--no_file_tests",
            "--gtest_filter=" + test_suite + ".*",
            "--fail_if_no_tests_run",
        ],
    )
    for (test_dir, test_target, test_suite) in EXPLORER_GTEST_TESTS
]

glob_sh_run(
    args = ["$(location //explorer)"],
    data = ["//explorer"],
    file_exts = ["carbon"],
)

glob_sh_run(
    args = [
        "$(location //explorer)",
        "--parser_debug",
        "--trace_file=-",
    ],
    data = ["//explorer"],
    file_exts = ["carbon"],
    run_ext = "verbose",
)

filegroup(
    name = "carbon_files",
    srcs = glob(["testdata/**/*.carbon"]),
    # Files are used for validating fuzzer completeness.
    visibility = ["//explorer/fuzzing:__pkg__"],
)

filegroup(
    name = "treesitter_testdata",
    srcs = glob(
        ["testdata/**/*.carbon"],
        exclude = [
            "testdata/**/fail_*",
        ],
    ),
    visibility = ["//utils/treesitter:__pkg__"],
)
