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

# All explorer tests are packaged into a single binary to reduce disk usage.
cc_binary(
    name = "explorer_tests",
    testonly = 1,
    srcs = ["file_test.cpp"],
    deps = [
        ":main",
        "//common:check",
        "//explorer/ast:ast_test_matchers_test",
        "//explorer/ast:expression_test",
        "//explorer/ast:element_test",
        "//explorer/ast:pattern_test",
        "//explorer/base:arena_test",
        "//explorer/base:decompose_test",
        "//explorer/base:error_builders_test",
        "//explorer/base:set_file_context_raii_test",
        "//explorer/base:set_program_phase_raii_test",
        "//explorer/parse_and_execute:parse_and_execute_test",
        "//explorer/syntax:parse_test",
        "//explorer/syntax:unimplemented_example_test",
        "//testing/base:test_raw_ostream",
        "//testing/file_test:file_test_base",
        "@com_google_absl//absl/flags:flag",
        "@com_googlesource_code_re2//:re2",
    ],
)

file_test(
    name = "file_test",
    prebuilt_binary = ":explorer_tests",
    size = "small",
    args = ["--gtest_filter=ExplorerFileTest.*"],
    shard_count = 20,
    tests = glob(["testdata/**/*.carbon"]),
)

file_test(
    name = "file_test.trace",
    prebuilt_binary = ":explorer_tests",
    size = "small",
    args = ["--trace", "--gtest_filter=ExplorerFileTest.*"],
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

sh_test(
    name = "ast_test_matchers_test",
    srcs = [":explorer_tests"],
    args = ["--no_file_tests", "--gtest_filter=ASTTestMatchers.*"],
    size = "small",
)

sh_test(
    name = "ast_expression_test",
    srcs = [":explorer_tests"],
    args = ["--no_file_tests", "--gtest_filter=ExpressionTest.*"],
    size = "small",
)

sh_test(
    name = "ast_element_test",
    srcs = [":explorer_tests"],
    args = ["--no_file_tests", "--gtest_filter=ElementTest.*"],
    size = "small",
)

sh_test(
    name = "ast_pattern_test",
    srcs = [":explorer_tests"],
    args = ["--no_file_tests", "--gtest_filter=PatternTest.*"],
    size = "small",
)

sh_test(
    name = "base_arena_test",
    srcs = [":explorer_tests"],
    args = ["--no_file_tests", "--gtest_filter=ArenaTest.*"],
    size = "small",
)

sh_test(
    name = "base_decompose_test",
    srcs = [":explorer_tests"],
    args = ["--no_file_tests", "--gtest_filter=HashFromDecomposeTest.*"],
    size = "small",
)

sh_test(
    name = "base_error_builders_test",
    srcs = [":explorer_tests"],
    args = ["--no_file_tests", "--gtest_filter=ErrorBuildersTest.*"],
    size = "small",
)

sh_test(
    name = "base_set_file_context_raii_test",
    srcs = [":explorer_tests"],
    args = ["--no_file_tests", "--gtest_filter=SetFileContextRaiiTest.*"],
    size = "small",
)

sh_test(
    name = "base_set_program_phase_raii_test",
    srcs = [":explorer_tests"],
    args = ["--no_file_tests", "--gtest_filter=SetProgramPhaseRaiiTest.*"],
    size = "small",
)

sh_test(
    name = "parse_and_execute_test",
    srcs = [":explorer_tests"],
    args = ["--no_file_tests", "--gtest_filter=ParseAndExecuteTest.*"],
    size = "small",
)

sh_test(
    name = "syntax_parse_test",
    srcs = [":explorer_tests"],
    args = ["--no_file_tests", "--gtest_filter=ParseTest.*"],
    size = "small",
)

sh_test(
    name = "syntax_unimplemented_example_test",
    srcs = [":explorer_tests"],
    args = ["--no_file_tests", "--gtest_filter=UnimplementedExampleTest.*"],
    size = "small",
)

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
