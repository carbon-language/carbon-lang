# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_bison//bison:bison.bzl", "bison_cc_library")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_flex//flex:flex.bzl", "flex")

bison_cc_library(
  name = "syntax_bison",
  src = "syntax.y",
)

flex(
  name = "syntax_flex",
  src = "syntax.l",
)
