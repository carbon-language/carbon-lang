// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver_subcommand.h"

#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "toolchain/driver/lld_runner.h"

namespace Carbon {

auto DriverSubcommand::DisableFuzzingExternalLibraries(DriverEnv& driver_env,
                                                       llvm::StringRef name)
    -> bool {
  // Only need to do anything when fuzzing.
  if (!driver_env.fuzzing) {
    return true;
  }

  CARBON_DIAGNOSTIC(
      ToolFuzzingDisallowed, Error,
      "preventing fuzzing of `{0}` subcommand due to external library",
      std::string);
  driver_env.emitter.Emit(ToolFuzzingDisallowed, name.str());
  return false;
}

}  // namespace Carbon
