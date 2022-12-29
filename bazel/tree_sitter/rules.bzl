# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for building fuzz tests."""

def tree_sitter(
        name,
        grammar):
    tools = ["@tree_sitter_bin//:tree-sitter", "@nodejs_host//:node_bin"]

    # This only saves parser.c because it's all that's really needed for C++.
    # Outputs are listed at:
    # https://tree-sitter.github.io/tree-sitter/creating-parsers#command-generate
    parser_c = "{0}.c".format(name)
    native.genrule(
        name = "{0}.generate".format(name),
        tools = tools,
        srcs = [grammar],
        cmd = "PATH=$$PATH:$$(dirname $(location @nodejs_host//:node_bin)) " +
              "$(location @tree_sitter_bin//:tree-sitter) generate $<; " +
              "mv src/parser.c $@",
        outs = [parser_c],
    )
    native.cc_library(
        name = name,
        srcs = [parser_c],
        deps = ["@tree_sitter//:tree_sitter"],
    )

    # Allow running tree-sitter directly with `name.tree-sitter`.
    tree_sitter_sh = "{0}.tree-sitter.sh".format(name)
    native.genrule(
        name = "{0}.genrule".format(tree_sitter_sh),
        cmd = "echo '$(location @tree_sitter_bin//:tree-sitter) \"$$@\"' > $@",
        outs = [tree_sitter_sh],
        tools = tools,
    )
    native.sh_binary(
        name = "{0}.tree-sitter".format(name),
        srcs = [tree_sitter_sh],
        data = tools,
    )
