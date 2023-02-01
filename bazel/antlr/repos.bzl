# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_jar")

_version = "4.11.1"

def antlr_repos():
    """Configures repos for ANTLR's tool and runtimes."""
    http_jar(
        name = "antlr4_tool",
        urls = ["https://www.antlr.org/download/antlr-{0}-complete.jar".format(_version)],
        sha256 = "62975e192b4af2622b72b5f0131553ee3cbce97f76dc2a41632dcc55e25473e1",
    )
    http_archive(
        name = "antlr4_runtimes",
        build_file = "//bazel/antlr:BUILD.runtimes",
        sha256 = "50e87636a61daabd424d884c60f804387430920072f585a9fee2b90e2043fdcc",
        strip_prefix = "antlr4-{0}".format(_version),
        urls = ["https://github.com/antlr/antlr4/archive/v{0}.tar.gz".format(_version)],
    )
