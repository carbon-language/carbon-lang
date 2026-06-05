// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_BUILD_SUBCOMMAND_H_
#define CARBON_TOOLCHAIN_DRIVER_BUILD_SUBCOMMAND_H_

#include "common/command_line.h"
#include "toolchain/driver/compile_options.h"
#include "toolchain/driver/driver_env.h"
#include "toolchain/driver/driver_subcommand.h"
#include "toolchain/driver/link_options.h"

namespace Carbon {

// Options for the build subcommand.
struct BuildSubcommandOptions {
  auto Build(CommandLine::CommandBuilder& b) -> void;

  CompileOptions compile_options;
  LinkOptions link_options;
  bool use_temp_dir;
};

class BuildSubcommand : public DriverSubcommand {
 public:
  explicit BuildSubcommand();

  auto BuildOptions(CommandLine::CommandBuilder& b) -> void override;

  auto Run(DriverEnv& driver_env) -> DriverResult override;

 private:
  BuildSubcommandOptions options_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_BUILD_SUBCOMMAND_H_
