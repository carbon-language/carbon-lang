// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/clang_subcommand.h"

#include "llvm/TargetParser/Host.h"
#include "toolchain/driver/clang_runner.h"

namespace Carbon {

constexpr CommandLine::CommandInfo ClangOptions::Info = {
    .name = "clang",
    .help = R"""(
Runs Clang on arguments.

This is equivalent to running the `clang` command line directly, and provides
the full command line interface.

Use `carbon clang -- ARGS` to pass flags to `clang`. Although there are
currently no flags for `carbon clang`, the `--` reserves the ability to add
flags in the future.

This is provided to help guarantee consistent compilation of C++ files, both
when Clang is invoked directly and when a Carbon file importing a C++ file
results in an indirect Clang invocation.
)""",
};

auto ClangOptions::Build(CommandLine::CommandBuilder& b) -> void {
  b.AddStringPositionalArg(
      {
          .name = "ARG",
          .help = R"""(
Arguments passed to Clang.
)""",
      },
      [&](auto& arg_b) { arg_b.Append(&args); });
}

// TODO: This lacks a lot of features from the main driver code. We may need to
// add more.
// https://github.com/llvm/llvm-project/blob/main/clang/tools/driver/driver.cpp
auto ClangSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  std::string target = llvm::sys::getDefaultTargetTriple();
  ClangRunner runner(driver_env.installation, target, driver_env.vlog_stream);
  return {.success = runner.Run(options_.args)};
}

}  // namespace Carbon
