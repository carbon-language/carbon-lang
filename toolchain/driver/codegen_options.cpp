// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/codegen_options.h"

namespace Carbon {

auto CodegenOptions::Build(CommandLine::CommandBuilder& b) -> void {
  b.AddStringOption(
      {
          .name = "target",
          .help = R"""(
Select a target platform. Uses the LLVM target syntax. Also known as a "triple"
for historical reasons.

This corresponds to the `target` flag to Clang and accepts the same strings
documented there:
https://clang.llvm.org/docs/CrossCompilation.html#target-triple
)""",
      },
      [&](auto& arg_b) {
        arg_b.Default(host);
        arg_b.Set(&target);
      });

  // TODO: We probably need a better infrstructure for forwarding flags to LLVM.
  b.AddFlag(
      {
          .name = "pie",
          .help = R"""(
Toggles PIE output in LLVM.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Default(true);
        arg_b.Set(&pie);
      });
}

}  // namespace Carbon
