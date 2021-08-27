# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//executable_semantics:__pkg__"])

cc_library(
    name = "bison_wrap",
    hdrs = ["bison_wrap.h"],
    deps = ["//common:check"],
)

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
        "//executable_semantics/common:tracing_flag",
        "//executable_semantics/interpreter",
        "//executable_semantics/interpreter:typecheck",
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
