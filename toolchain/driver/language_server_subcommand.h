// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_LANGUAGE_SERVER_SUBCOMMAND_H_
#define CARBON_TOOLCHAIN_DRIVER_LANGUAGE_SERVER_SUBCOMMAND_H_

#include "common/command_line.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/driver/codegen_options.h"
#include "toolchain/driver/driver_env.h"
#include "toolchain/driver/driver_subcommand.h"

namespace Carbon {

// Implements the link subcommand of the driver.
class LanguageServerSubcommand : public DriverSubcommand {
 public:
  explicit LanguageServerSubcommand();

  auto BuildOptions(CommandLine::CommandBuilder& /*b*/) -> void override {}

  auto Run(DriverEnv& driver_env) -> DriverResult override;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_LANGUAGE_SERVER_SUBCOMMAND_H_
