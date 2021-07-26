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
        "//executable_semantics/common:tracing_flag",
        "//executable_semantics/syntax",
        "@llvm-project//llvm:Support",
    ],
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
    "generic_function1",
    "generic_function2",
    "generic_function3",
    "generic_function_apply",
    "generic_function_fail1",
    "generic_function_fail2",
    "generic_function_fail3",
    "generic_function_swap",
    "generic_function_tuple_map",
    "global_variable1",
    "global_variable2",
    "global_variable3",
    "global_variable4",
    "global_variable5",
    "global_variable6",
    "global_variable7",
    "global_variable8",
    "ignored_parameter",
    "if1",
    "if2",
    "if3",
    "invalid_char",
    "match_any_int",
    "match_int_default",
    "match_int",
    "match_placeholder",
    "match_type",
    "next",
    "pattern_init",
    "pattern_variable_fail",
    "placeholder_variable",
    "record1",
    "star",
    "struct1",
    "struct2",
    "struct3",
    "struct_field_access_mismatch",
    "struct_field_mismatch",
    "struct_field_missing",
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
) for e in EXAMPLES]

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
) for e in EXAMPLES]
