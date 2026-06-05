// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/compile_subcommand.h"

#include "toolchain/driver/compile_driver.h"

namespace Carbon {

static constexpr CommandLine::CommandInfo SubcommandInfo = {
    .name = "compile",
    .help = R"""(
Compile Carbon source code.

This subcommand runs the Carbon compiler over input source code, checking it for
errors and producing the requested output.

Error messages are written to the standard error stream.

Different phases of the compiler can be selected to run, and intermediate state
can be written to standard output as these phases progress.
)""",
};

CompileSubcommand::CompileSubcommand() : DriverSubcommand(SubcommandInfo) {}

auto CompileSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  if (driver_env.fuzzing && !options_.clang_args.empty()) {
    // Parsing specific Clang arguments can reach deep into
    // external libraries that aren't fuzz clean.
    TestAndDiagnoseIfFuzzingExternalLibraries(driver_env, "compile");
    return {.success = false};
  }

  auto compile_driver = CompileDriver(&options_);

  if (!compile_driver.Initialize(driver_env,
                                 [&](llvm::StringRef) -> std::string {
                                   return options_.output_filename.str();
                                 })) {
    return {.success = false};
  }

  return compile_driver.Compile(driver_env);
}

}  // namespace Carbon
