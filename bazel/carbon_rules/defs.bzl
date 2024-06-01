# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provides rules for building Carbon files using the toolchain."""

load("@bazel_skylib//rules:run_binary.bzl", "run_binary")

def carbon_binary(name, srcs):
    """Compiles a Carbon binary.

    Args:
      name: The name of the build target.
      srcs: List of Carbon source files to compile.
    """
    for src in srcs:
        # Build each source file. For now, we pass all sources to each compile
        # because we don't have visibility into dependencies and have no way to
        # specify multiple output files. Object code for each input is written
        # into the output file in turn, so the final carbon source file
        # specified ends up determining the contents of the object file.
        #
        # TODO: This is a hack; replace with something better once the toolchain
        # supports doing so.
        #
        # TODO: Switch to the `prefix_root` based rule similar to linking when
        # the prelude moves there.
        out = src + ".o"
        srcs_reordered = [s for s in srcs if s != src] + [src]
        run_binary(
            name = src + ".compile",
            tool = "//toolchain/driver:carbon",
            args = (["compile"] +
                    ["$(location %s)" % s for s in srcs_reordered] +
                    ["--output=$(location %s)" % out]),
            srcs = srcs,
            outs = [out],
        )

    # For now, we assume that the prelude doesn't produce any necessary object
    # code, and don't include the .o files for //core/prelude... in the final
    # linked binary.
    #
    # TODO: This will need to be revisited eventually.
    objs = [s + ".o" for s in srcs]
    native.genrule(
        name = name + ".link",
        tools = [
            "//toolchain/install:prefix_root/bin/carbon",
            "//toolchain/install:install_data",
        ],
        cmd = "$(execpath //toolchain/install:prefix_root/bin/carbon) link --output=$@ $(SRCS)",
        srcs = objs,
        outs = [name],
    )
