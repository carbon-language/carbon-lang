# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@mypy_integration//:mypy.bzl", "mypy_test")

package(default_visibility = ["//executable_semantics:__pkg__"])

cc_library(
    name = "bison_wrap",
    hdrs = ["bison_wrap.h"],
    deps = ["//common:check"],
)

cc_test(
    name = "parse_test",
    srcs = ["parse_test.cpp"],
    deps = [
        ":syntax",
        "//executable_semantics/common:arena",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "parse_test_matchers",
    testonly = 1,
    srcs = ["parse_test_matchers_internal.h"],
    hdrs = ["parse_test_matchers.h"],
    deps = [
        ":syntax",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "syntax",
    srcs = [
        "lexer.cpp",
        "lexer.h",
        "parse.cpp",
        "parse_and_lex_context.cpp",
        "parse_and_lex_context.h",
        "parser.cpp",
        "parser.h",
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
        ":bison_wrap",
        "//common:check",
        "//common:ostream",
        "//common:string_helpers",
        "//executable_semantics/ast",
        "//executable_semantics/ast:declaration",
        "//executable_semantics/ast:expression",
        "//executable_semantics/ast:paren_contents",
        "//executable_semantics/common:arena",
        "//executable_semantics/common:error",
    ],
)

genrule(
    name = "syntax_bison_srcs",
    srcs = ["parser.ypp"],
    outs = [
        "parser.cpp",
        "parser.h",
        "parser.output",
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
    outs = [
        "lexer.cpp",
        "lexer.h",
    ],
    cmd = "M4=$(M4) $(FLEX) " +
          "--outfile=$(location lexer.cpp) " +
          "--header-file=$(location lexer.h) " +
          "$(location lexer.lpp)",
    toolchains = [
        "@rules_flex//flex:current_flex_toolchain",
        "@rules_m4//m4:current_m4_toolchain",
    ],
)

py_library(
    name = "format_grammar_lib",
    srcs = ["format_grammar.py"],
)

py_test(
    name = "format_grammar_test",
    srcs = ["format_grammar_test.py"],
    deps = ["format_grammar_lib"],
)

mypy_test(
    name = "format_grammar_mypy_test",
    include_imports = True,
    deps = [":format_grammar_lib"],
)

cc_test(
    name = "unimplemented_example_test",
    srcs = ["unimplemented_example_test.cpp"],
    deps = [
        ":parse_test_matchers",
        ":syntax",
        "//executable_semantics/ast:ast_test_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
