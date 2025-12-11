# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A Starlark implementation of a CMake-like configure_file rule."""

def _configure_cmake_file_impl(ctx):
    """Implementation for the configure_cmake_file rule."""

    # Flatten the defines dictionary into a list of command-line arguments
    # for the implementation script:
    #
    #   ["--defines", KEY1, VAL1, "--defines", KEY2, VAL2]
    define_args = []
    for key, value in ctx.attr.defines.items():
        define_args.append("--defines")
        define_args.append(key)
        define_args.append(value)

    ctx.actions.run(
        executable = ctx.executable._impl_script,
        arguments = [
            "--src",
            ctx.file.src.path,
            "--out",
            ctx.outputs.out.path,
        ] + define_args,
        inputs = depset([ctx.file.src, ctx.executable._impl_script]),
        outputs = [ctx.outputs.out],
        mnemonic = "ConfigureCmakeFile",
        progress_message = "Configuring file: %{label}",
    )

    return [DefaultInfo(files = depset([ctx.outputs.out]))]

configure_cmake_file = rule(
    implementation = _configure_cmake_file_impl,
    attrs = {
        "defines": attr.string_dict(
            mandatory = True,
            doc = "A dictionary of key-value definitions to substitute.",
        ),
        "out": attr.output(
            mandatory = True,
            doc = "The generated output file.",
        ),
        "src": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "The input '.in' template file.",
        ),
        "_impl_script": attr.label(
            default = Label("//toolchain/install:configure_cmake_file_impl"),
            allow_files = True,
            executable = True,
            cfg = "exec",
        ),
    },
    doc = """
A rule that performs CMake-style configuration of an input file.

This rule processes an input file (`.in`) and generates an output file
based on a dictionary of definitions. It provides emulation
of the most commonly used aspects of CMake's `configure_file` command:
https://cmake.org/cmake/help/latest/command/configure_file.html

Notable aspects not implemented are the following:

*   Substitution of cache values using `$CACHE{VAR}` syntax.
*   Substitution of environment variables using `$ENV{VAR}` syntax.
""",
)
