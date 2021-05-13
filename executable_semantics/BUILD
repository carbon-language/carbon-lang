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
    "assignment_copy1",
    "assignment_copy2",
    "block1",
    "block2",
    "break1",
    "choice1",
    "continue1",
    "fun_named_params",
    "fun_named_params2",
    "fun_recur",
    "fun1",
    "fun2",
    "fun3",
    "fun4",
    "fun5",
    "fun6_fail_type",
    "funptr1",
    "global_variable1",
    "global_variable2",
    "global_variable3",
    "global_variable4",
    "global_variable5",
    "global_variable6",
    "global_variable7",
    "global_variable8",
    "if1",
    "if2",
    "if3",
    "match_any_int",
    "match_int_default",
    "match_int",
    "match_type",
    "next",
    "pattern_init",
    "pattern_variable_fail",
    "record1",
    "star",
    "struct1",
    "struct2",
    "struct3",
    "tuple_assign",
    "tuple_equality",
    "tuple_equality2",
    "tuple_equality3",
    "tuple_match",
    "tuple_match2",
    "tuple_match3",
    "tuple1",
    "tuple2",
    "tuple3",
    "tuple4",
    "tuple5",
    "type_compute",
    "type_compute2",
    "type_compute3",
    "while1",
    "zero",
    "experimental_continuation1",
    "experimental_continuation2",
    "experimental_continuation3",
    "experimental_continuation4",
    "experimental_continuation5",
    "experimental_continuation6",
    "experimental_continuation7",
    "experimental_continuation8",
    "experimental_continuation9",
]

[genrule(
    name = "%s_out" % e,
    srcs = ["testdata/%s.carbon" % e],
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
