# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//executable_semantics:__subpackages__"])

cc_library(
    name = "syntax",
    srcs = [
        "lexer.cpp",
        "parse.cpp",
        "parse_and_lex_context.cpp",
        "parse_and_lex_context.h",
        "parser.cpp",
        "parser.h",
        "syntax_helpers.cpp",
        "syntax_helpers.h",
    ],
    hdrs = [
        "parse.h",
    ],
    # Disable warnings for generated code.
    copts = [
        "-Wno-unneeded-internal-declaration",
        "-Wno-unused-function",
        "-Wno-writable-strings",
    ],
    deps = [
        ":paren_contents",
        "//common:check",
        "//common:ostream",
        "//executable_semantics/ast:declaration",
        "//executable_semantics/ast:expression",
        "//executable_semantics/common:arena",
        "//executable_semantics/common:error",
        "//executable_semantics/common:tracing_flag",
        "//executable_semantics/interpreter",
        "//executable_semantics/interpreter:typecheck",
    ],
)

cc_library(
    name = "paren_contents",
    srcs = ["paren_contents.cpp"],
    hdrs = ["paren_contents.h"],
    deps = ["//executable_semantics/ast:expression"],
)

cc_test(
    name = "paren_contents_test",
    srcs = ["paren_contents_test.cpp"],
    env = {
        # TODO(#580): Remove this when leaks are fixed.
        "ASAN_OPTIONS": "detect_leaks=0",
    },
    deps = [
        ":paren_contents",
        "@llvm-project//llvm:gtest",
        "@llvm-project//llvm:gtest_main",
    ],
)

genrule(
    name = "syntax_bison_srcs",
    srcs = ["parser.ypp"],
    outs = [
        "parser.cpp",
        "parser.h",
    ],
    cmd = "M4=$(M4) $(BISON) " +
          "--output=$(location parser.cpp) " +
          "--report=state " +
          "--defines=$(location parser.h) " +
          "$(location parser.ypp)",
    toolchains = [
        "@rules_bison//bison:current_bison_toolchain",
        "@rules_m4//m4:current_m4_toolchain",
    ],
)

genrule(
    name = "syntax_flex_srcs",
    srcs = ["lexer.lpp"],
    outs = ["lexer.cpp"],
    cmd = "M4=$(M4) $(FLEX) " +
          "--outfile=$(location lexer.cpp) " +
          "$(location lexer.lpp)",
    toolchains = [
        "@rules_flex//flex:current_flex_toolchain",
        "@rules_m4//m4:current_m4_toolchain",
    ],
)
