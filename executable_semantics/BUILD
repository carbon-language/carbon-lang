# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_binary(
    name = "executable_semantics",
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
