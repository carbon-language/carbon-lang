// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/link_options.h"

namespace Carbon {

namespace {

auto BuildSharedOptions(CommandLine::CommandBuilder& b, LinkOptions* options)
    -> void {
  b.AddStringPositionalArg(
      {
          .name = "EXTRA_CLANG_LINK_ARGS",
          .help = R"""(
Extra arguments to pass to Clang when forming the link command. This is
primarily useful for expanding `LDFLAGS` or other baseline linking flags in a
build system.

These can also be used to pass object files to the link in the event your build
system mixes object files and linker flags.
)""",
      },
      [&](auto& arg_b) { arg_b.Append(&options->extra_clang_args); });
}

}  // namespace

auto LinkOptions::BuildForLinkSubcommand(CommandLine::CommandBuilder& b)
    -> void {
  b.AddStringPositionalArg(
      {
          .name = "OBJECT_FILE",
          .help = R"""(
The input object files.

If empty, there must be extra Clang link arguments that provide object files
intermingled with linking flags.
)""",
      },
      [&](auto& arg_b) { arg_b.Append(&object_filenames); });

  b.AddStringOption(
      {
          .name = "output",
          .value_name = "FILE",
          .help = R"""(
The linked file name. The output is always a linked binary.

If not provided, there must be extra Clang link arguments that include
specifying the output of the link. This allows supporting build systems that
intermingle the output flag with arbitrary other linker flags that need to use
legacy parsing logic.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&output_filename); });

  BuildSharedOptions(b, this);

  codegen_options = std::make_shared<CodegenOptions>();
  codegen_options->Build(b);
}

auto LinkOptions::BuildForBuildSubcommand(CommandLine::CommandBuilder& b)
    -> void {
  b.AddStringOption(
      {
          .name = "output",
          .short_name = "o",
          .value_name = "FILE",
          .help = R"""(
The file name for the output binary. If none is specified, `build` will use the
name of the first provided input file.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&output_filename); });

  BuildSharedOptions(b, this);
}

}  // namespace Carbon
