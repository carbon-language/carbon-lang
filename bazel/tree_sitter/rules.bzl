# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for building fuzz tests."""

def tree_sitter(
        name,
        grammar):
    # This only saves parser.c because it's all that's really needed for C++.
    # Outputs are listed at:
    # https://tree-sitter.github.io/tree-sitter/creating-parsers#command-generate
    outs = [
        ("src/parser.c", "{0}/tree_sitter/parser.c".format(name)),
        ("src/tree_sitter/parser.h", "{0}/tree_sitter/parser.h".format(name)),
    ]
    native.genrule(
        name = "{0}.parser".format(name),
        tools = ["@tree_sitter_bin//:tree-sitter", "@nodejs_host//:node_bin"],
        srcs = [grammar],
        cmd = "PATH=$$PATH:$$(dirname $(location @nodejs_host//:node_bin)) $(location @tree_sitter_bin//:tree-sitter) generate $<; " +
              "mv src/tree_sitter/parser.h $@",
        outs = ["{0}.c".format(name)],
    )
    #cc_library(name = name, src = [])
