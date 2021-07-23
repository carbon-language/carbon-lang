# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO(https://github.com/carbon-language/carbon-lang/issues/266):
# Migrate bison/flex usage to a more hermetic bazel build.

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("//bazel/testing:golden_test.bzl", "golden_test")
load("test_list.bzl", "TEST_LIST")

cc_binary(
    name = "executable_semantics",
    srcs = ["main.cpp"],
    deps = [
        "//executable_semantics/common:tracing_flag",
        "//executable_semantics/syntax",
        "@llvm-project//llvm:Support",
    ],
)

[golden_test(
    name = "%s_test" % e,
    cmd = "'$(location executable_semantics) $(location testdata/%s.carbon)'" % e,
    data = [
        ":executable_semantics",
        "testdata/%s.carbon" % e,
    ],
    env = {
        # TODO(#580): Remove this when leaks are fixed.
        "ASAN_OPTIONS": "detect_leaks=0",
    },
    golden = "testdata/%s.golden" % e,
) for e in TEST_LIST]

# Test --trace by expecting golden output to be a *subset* of trace output. Note
# the normal test must be used to update golden files.
[golden_test(
    name = "%s_trace_test" % e,
    cmd = "'$(location executable_semantics) --trace " +
          "$(location testdata/%s.carbon)'" % e,
    data = [
        ":executable_semantics",
        "testdata/%s.carbon" % e,
    ],
    env = {
        # TODO(#580): Remove this when leaks are fixed.
        "ASAN_OPTIONS": "detect_leaks=0",
    },
    golden = "testdata/%s.golden" % e,
    golden_is_subset = True,
) for e in TEST_LIST]
