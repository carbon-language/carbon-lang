// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_DRIVER_SUBCOMMAND_H_
#define CARBON_TOOLCHAIN_DRIVER_DRIVER_SUBCOMMAND_H_

#include "common/command_line.h"
#include "common/ostream.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "toolchain/driver/driver_env.h"
#include "toolchain/install/install_paths.h"

namespace Carbon {

// The result of a driver run.
struct DriverResult {
  // Overall success result.
  bool success;

  // Per-file success results. May be empty if files aren't individually
  // processed.
  llvm::SmallVector<std::pair<std::string, bool>> per_file_success = {};
};

// A subcommand for the driver.
class DriverSubcommand {
 public:
  explicit DriverSubcommand(CommandLine::CommandInfo info) : info_(info) {}

  // Adds the subcommand to the main command, assigning `selected_command` when
  // the subcommand is in use.
  auto AddTo(CommandLine::CommandBuilder& b,
             DriverSubcommand** selected_subcommand) -> void {
    b.AddSubcommand(info_, [this, selected_subcommand](
                               CommandLine::CommandBuilder& sub_b) {
      BuildOptions(sub_b);
      sub_b.Do([this, selected_subcommand] { *selected_subcommand = this; });
    });
  }

  // Adds command line options.
  virtual auto BuildOptions(CommandLine::CommandBuilder& b) -> void = 0;

  // Runs the command.
  virtual auto Run(DriverEnv& driver_env) -> DriverResult = 0;

 private:
  // Subcommand information.
  CommandLine::CommandInfo info_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_DRIVER_SUBCOMMAND_H_
