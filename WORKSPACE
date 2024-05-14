# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

workspace(name = "carbon")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

###############################################################################
# Example conversion repositories
###############################################################################

local_repository(
    name = "brotli",
    path = "third_party/examples/brotli/original",
)

new_local_repository(
    name = "woff2",
    build_file = "third_party/examples/woff2/BUILD.original",
    path = "third_party/examples/woff2/original",
    workspace_file = "third_party/examples/woff2/WORKSPACE.original",
)

local_repository(
    name = "woff2_carbon",
    path = "third_party/examples/woff2/carbon",
)

###############################################################################
# Treesitter rules
###############################################################################

http_archive(
    name = "rules_nodejs",
    sha256 = "d124665ea12f89153086746821cf6c9ef93ab88360a50c1aeefa1fe522421704",
    strip_prefix = "rules_nodejs-6.0.0-beta1",
    url = "https://github.com/bazelbuild/rules_nodejs/releases/download/v6.0.0-beta1/rules_nodejs-v6.0.0-beta1.tar.gz",
)

load("@rules_nodejs//nodejs:repositories.bzl", "DEFAULT_NODE_VERSION", "nodejs_register_toolchains")

nodejs_register_toolchains(
    name = "nodejs",
    node_version = DEFAULT_NODE_VERSION,
)

http_archive(
    name = "rules_tree_sitter",
    sha256 = "a09f177a2b8acb2f8a84def6ca0c41a5bd26b25634aa7313f22ade6c54e57ca1",
    strip_prefix = "rules_tree_sitter-bc3a2131053207de7dfd9b24046b811ce770e35d",
    urls = ["https://github.com/Maan2003/rules_tree_sitter/archive/bc3a2131053207de7dfd9b24046b811ce770e35d.tar.gz"],
)

load("@rules_tree_sitter//tree_sitter:tree_sitter.bzl", "tree_sitter_register_toolchains")

tree_sitter_register_toolchains()
