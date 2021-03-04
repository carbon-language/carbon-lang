# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO(https://github.com/carbon-language/carbon-lang/issues/266):
# Migrate bison/flex usage to a more hermetic bazel build.

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//executable_semantics:__subpackages__"])

cc_library(
    name = "syntax",
    srcs = [
        "syntax.tab.cpp",
        "syntax.yy.cpp",
        "syntax_helpers.cpp",
        "syntax_helpers.h",
    ],
    hdrs = ["syntax.tab.h"],
    # Disable warnings for generated code.
    copts = [
        "-Wno-unneeded-internal-declaration",
        "-Wno-unused-function",
        "-Wno-writable-strings",
    ],
    deps = [
        "//executable_semantics:tracing_flag",
        "//executable_semantics/ast:declaration",
        "//executable_semantics/ast:expression",
        "//executable_semantics/ast:field_list",
        "//executable_semantics/interpreter",
    ],
)

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
