# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# ANTLR's expected extension.
_grammar_ext = ".g4"

def antlr_cc_library(name, src, package):
    """Creates a C++ lexer and parser from a source grammar.

    Args:
      name: Base name for the lexer and the parser rules.
      src: ANTLR grammar file.
      package: The namespace for the generated code.
    """
    generated = "{0}_grammar".format(name)
    antlr_library(
        name = generated,
        src = src,
        package = package,
    )
    native.cc_library(
        name = name,
        srcs = [generated],
        deps = ["@antlr4_runtimes//:cpp"],
        copts =
            [
                "-Wno-logical-op-parentheses",
                "-Wno-unused-parameter",
            ],
        linkstatic = 1,
    )

def _antlr_library(ctx):
    output = ctx.actions.declare_directory(ctx.attr.name)

    antlr_args = ctx.actions.args()
    antlr_args.add("-Dlanguage=Cpp")
    antlr_args.add("-no-listener")
    antlr_args.add("-visitor")
    antlr_args.add("-o", output.path)
    antlr_args.add("-package", ctx.attr.package)
    antlr_args.add(ctx.file.src)

    ctx.actions.run(
        arguments = [antlr_args],
        inputs = [ctx.file.src],
        outputs = [output],
        executable = ctx.executable._tool,
        progress_message = "Processing ANTLR grammar",
    )

    files = []
    generated_base = "{0}/{1}".format(
        output.path,
        ctx.file.src.path.removesuffix(_grammar_ext),
    )
    out_base = ctx.file.src.basename.removesuffix(_grammar_ext)

    for suffix in ("Lexer", "Parser", "BaseVisitor", "Visitor"):
        mnemonic_suffix = "Copy{0}".format(suffix)
        generated_suffix = "{0}{1}.".format(generated_base, suffix)
        out_suffix = "{0}{1}.".format(out_base, suffix)

        for ext in ("h", "cpp"):
            f = ctx.actions.declare_file(out_suffix + ext)
            ctx.actions.run_shell(
                mnemonic = mnemonic_suffix + ext,
                inputs = [output],
                outputs = [f],
                command = 'cp "{0}" "{1}"'.format(generated_suffix + ext, f.path),
            )
            files.append(f)

    compilation_context = cc_common.create_compilation_context(
        headers = depset(files),
    )
    return [
        DefaultInfo(files = depset(files)),
        CcInfo(compilation_context = compilation_context),
    ]

antlr_library = rule(
    implementation = _antlr_library,
    attrs = {
        "src": attr.label(allow_single_file = [_grammar_ext], mandatory = True),
        "package": attr.string(),
        "_tool": attr.label(
            executable = True,
            cfg = "host",  # buildifier: disable=attr-cfg
            default = Label("//bazel/antlr:antlr4_tool"),
        ),
    },
)
