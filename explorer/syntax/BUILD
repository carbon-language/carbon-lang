# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//explorer/parse_and_execute:__pkg__"])

cc_library(
    name = "bison_wrap",
    hdrs = ["bison_wrap.h"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = ["//common:check"],
)

cc_test(
    name = "parse_test",
    size = "small",
    srcs = ["parse_test.cpp"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        ":syntax",
        "//explorer/base:arena",
        "//testing/base:gtest_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "parse_test_matchers",
    testonly = 1,
    srcs = ["parse_test_matchers_internal.h"],
    hdrs = ["parse_test_matchers.h"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        ":syntax",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "prelude",
    srcs = ["prelude.cpp"],
    hdrs = ["prelude.h"],
    data = ["//explorer:standard_libraries"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        ":syntax",
        "//common:error",
        "//explorer/ast",
        "//explorer/base:arena",
        "//explorer/base:nonnull",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "syntax",
    srcs = [
        "lex_helper.h",
        "lex_scan_helper.cpp",
        "lex_scan_helper.h",
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
        "-Wno-implicit-fallthrough",  # Needed to make yyinput() code compile.
        "-Wno-unneeded-internal-declaration",
        "-Wno-unused-but-set-variable",
        "-Wno-unused-function",
        "-Wno-writable-strings",
    ],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    visibility = [
        "//explorer/fuzzing:__pkg__",
        "//explorer/parse_and_execute:__pkg__",
    ],
    deps = [
        ":bison_wrap",
        "//common:check",
        "//common:error",
        "//common:ostream",
        "//common:string_helpers",
        "//explorer/ast",
        "//explorer/ast:expression_category",
        "//explorer/ast:paren_contents",
        "//explorer/base:arena",
        "//explorer/base:error_builders",
        "//explorer/base:nonnull",
        "//explorer/base:source_location",
        "@llvm-project//llvm:Support",
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
    size = "small",
    srcs = ["format_grammar_test.py"],
    deps = ["format_grammar_lib"],
)

cc_test(
    name = "unimplemented_example_test",
    size = "small",
    srcs = ["unimplemented_example_test.cpp"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        ":parse_test_matchers",
        ":syntax",
        "//explorer/ast:ast_test_matchers",
        "//testing/base:gtest_main",
        "@com_google_googletest//:gtest",
    ],
)
