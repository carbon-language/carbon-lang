# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("//bazel/testing:golden_test.bzl", "golden_test")

cc_binary(
    name = "executable_semantics",
    linkopts = [
        "-lpthread",
    ],
    deps = [":syntax"],
)

# TODO: It may be helpful to break this apart.
cc_library(
    name = "lib",
    srcs = [
        "ast.cpp",
        "interp.cpp",
        "typecheck.cpp",
    ],
    hdrs = [
        "assoc_list.h",
        "ast.h",
        "cons_list.h",
        "interp.h",
        "typecheck.h",
    ],
)

# TODO: Investigate further if this can use the @rules_flex `bison_cc_library` rule.
# load("@rules_bison//bison:bison.bzl", "bison_cc_library")
genrule(
    name = "syntax_bison_srcs",
    srcs = ["syntax.ypp"],
    outs = [
        "syntax.tab.cpp",
        "syntax.tab.h",
    ],
    cmd = "M4=$(M4) $(BISON) " +
          "--output=$(location syntax.tab.cpp) " +
          "--defines=$(location syntax.tab.h) " +
          "$(location syntax.ypp)",
    toolchains = [
        "@rules_bison//bison:current_bison_toolchain",
        "@rules_m4//m4:current_m4_toolchain",
    ],
)

cc_library(
    name = "syntax",
    srcs = [
        ":syntax.tab.cpp",
        ":syntax.yy.cpp",
    ],
    hdrs = [":syntax.tab.h"],
    # Disable warnings for generated code.
    copts = [
        "-Wno-unneeded-internal-declaration",
        "-Wno-unused-function",
        "-Wno-writable-strings",
    ],
    deps = [":lib"],
)

# TODO: Investigate further if this can use the @rules_flex `flex` rule.
# load("@rules_flex//flex:flex.bzl", "flex")
genrule(
    name = "syntax_flex_srcs",
    srcs = ["syntax.lpp"],
    outs = ["syntax.yy.cpp"],
    cmd = "M4=$(M4) $(FLEX) " +
          "--outfile=$(location syntax.yy.cpp) " +
          "$(location syntax.lpp)",
    toolchains = [
        "@rules_flex//flex:current_flex_toolchain",
        "@rules_m4//m4:current_m4_toolchain",
    ],
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
    "match_int_default",
    "match_int",
    "match_type",
    "next",
    "pattern_init",
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
    cmd = "$(location executable_semantics) $< > $@ 2>&1 || true",
    tools = [":executable_semantics"],
) for e in EXAMPLES]

[golden_test(
    name = "%s_test" % e,
    golden = "testdata/%s.golden" % e,
    subject = "testdata/%s.out" % e,
) for e in EXAMPLES]
