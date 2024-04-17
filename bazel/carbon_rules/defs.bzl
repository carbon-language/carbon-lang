# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provides rules for building Carbon files using the toolchain."""

load("@rules_cc//cc:defs.bzl", "cc_binary")

def carbon_binary(name, srcs):
    """Compiles a Carbon binary.

    Args:
      name: The name of the build target.
      srcs: List of Carbon source files to compile.
    """
    carbon_deps = ["//core:prelude", "//core:prelude_deps"] + srcs
    for src in srcs:
        # Build each source file. For now, we pass all sources to each compile
        # because we don't have visibility into dependencies and have no way to
        # specify multiple output files. Object code for each input is written
        # into the output file in turn, so the final carbon source file
        # specified ends up determining the contents of the object file.
        #
        # We also pass in the prelude files, but not prelude.carbon because the
        # driver adds that itself. For now, we assume that the prelude doesn't
        # produce any necessary object code, and don't include the .o files for
        # //core/prelude... in the final linked binary.
        #
        # TODO: This is a hack; replace with something better once the toolchain
        # supports doing so.
        inputs = ["//core:prelude_deps"] + [s for s in srcs if s != src] + [src]
        input_srcs = " ".join(["$(locations %s)" % input for input in inputs])
        native.genrule(
            name = src + ".compile",
            cmd = "$(location //toolchain/driver:carbon) compile %s --output=$(OUTS)" % input_srcs,
            srcs = carbon_deps,
            outs = [src + ".o"],
            tools = ["//toolchain/driver:carbon"],
        )
    cc_binary(name = name, srcs = [src + ".o" for src in srcs])
