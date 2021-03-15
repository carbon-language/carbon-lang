# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO(https://github.com/carbon-language/carbon-lang/issues/266):
# Migrate bison/flex usage to a more hermetic bazel build.

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("//bazel/testing:golden_test.bzl", "golden_test")

cc_binary(
    name = "executable_semantics",
    srcs = ["main.cpp"],
    deps = [
        "//executable_semantics/syntax",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "tracing_flag",
    srcs = ["tracing_flag.cpp"],
    hdrs = ["tracing_flag.h"],
    visibility = ["//executable_semantics:__subpackages__"],
)

EXAMPLES = [
    "block1",
    "break1",
    "choice1",
    "continue1",
    "fun_recur",
    "fun1",
    "fun2",
    "fun3",
    "fun4",
    "fun5",
    "fun6_fail_type",
    "funptr1",
    "if1",
    "if2",
    "if3",
    "match_int_default",
    "match_int",
    "match_type",
    "next",
    "pattern_init",
    "pattern_variable_fail",
    "record1",
    "struct1",
    "struct2",
    "struct3",
    "tuple_assign",
    "tuple_match",
    "tuple1",
    "tuple2",
    "undef1",
    "undef2",
    "while1",
    "zero",
]

[genrule(
    name = "%s_out" % e,
    srcs = ["testdata/%s.6c" % e],
    outs = ["testdata/%s.out" % e],
    # Suppress command errors.
    cmd = "$(location executable_semantics) $< > $@ 2>&1 || echo EXIT CODE: $$? >> $@",
    tools = [":executable_semantics"],
) for e in EXAMPLES]

[golden_test(
    name = "%s_test" % e,
    golden = "testdata/%s.golden" % e,
    subject = "testdata/%s.out" % e,
) for e in EXAMPLES]
