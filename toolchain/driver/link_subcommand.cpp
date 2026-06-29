// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/link_subcommand.h"

#include "toolchain/driver/link_driver.h"

namespace Carbon {

static constexpr CommandLine::CommandInfo SubcommandInfo = {
    .name = "link",
    .help = R"""(
Link Carbon executables.

This subcommand links Carbon executables by combining object files.

TODO: Support linking binary libraries, both archives and shared libraries.
)""",
};

LinkSubcommand::LinkSubcommand() : DriverSubcommand(SubcommandInfo) {}

auto LinkSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  // Don't run Clang when fuzzing, it is known to not be reliable under fuzzing
  // due to many unfixed issues.
  if (TestAndDiagnoseIfFuzzingExternalLibraries(driver_env, "clang")) {
    return {.success = false};
  }

  LinkDriver driver(&options_);
  return driver.Link(driver_env);
}

}  // namespace Carbon
