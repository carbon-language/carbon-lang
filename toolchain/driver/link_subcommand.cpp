// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/link_subcommand.h"

namespace Carbon {

constexpr CommandLine::CommandInfo LinkOptions::Info = {
    .name = "link",
    .help = R"""(
Link Carbon executables.

This subcommand links Carbon executables by combining object files.

TODO: Support linking binary libraries, both archives and shared libraries.
TODO: Support linking against binary libraries.
)""",
};

auto LinkOptions::Build(CommandLine::CommandBuilder& b,
                        CodegenOptions& codegen_options) -> void {
  b.AddStringPositionalArg(
      {
          .name = "OBJECT_FILE",
          .help = R"""(
The input object files.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Required(true);
        arg_b.Append(&object_filenames);
      });

  b.AddStringOption(
      {
          .name = "output",
          .value_name = "FILE",
          .help = R"""(
The linked file name. The output is always a linked binary.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Required(true);
        arg_b.Set(&output_filename);
      });

  codegen_options.Build(b);
}

}  // namespace Carbon
