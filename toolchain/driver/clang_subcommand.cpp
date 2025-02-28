// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/clang_subcommand.h"

#include "llvm/TargetParser/Host.h"
#include "toolchain/driver/clang_runner.h"

namespace Carbon {

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

static constexpr CommandLine::CommandInfo SubcommandInfo = {
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

ClangSubcommand::ClangSubcommand() : DriverSubcommand(SubcommandInfo) {}

// TODO: This lacks a lot of features from the main driver code. We may need to
// add more.
// https://github.com/llvm/llvm-project/blob/main/clang/tools/driver/driver.cpp
auto ClangSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  std::string target = llvm::sys::getDefaultTargetTriple();
  ClangRunner runner(driver_env.installation, target, driver_env.fs,
                     driver_env.vlog_stream);

  // Don't run Clang when fuzzing, it is known to not be reliable under fuzzing
  // due to many unfixed issues.
  if (driver_env.fuzzing) {
    CARBON_DIAGNOSTIC(
        ClangFuzzingDisallowed, Error,
        "preventing fuzzing of `clang` subcommand due to library crashes");
    driver_env.emitter.Emit(ClangFuzzingDisallowed);
    return {.success = false};
  }

  // Only enable Clang's leaking of memory if the driver can support that.
  if (driver_env.enable_leaking) {
    runner.EnableLeakingMemory();
  }

  return {.success = runner.Run(options_.args)};
}

}  // namespace Carbon
