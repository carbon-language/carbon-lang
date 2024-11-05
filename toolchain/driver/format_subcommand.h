// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_FORMAT_SUBCOMMAND_H_
#define CARBON_TOOLCHAIN_DRIVER_FORMAT_SUBCOMMAND_H_

#include "common/command_line.h"
#include "toolchain/driver/driver_env.h"
#include "toolchain/driver/driver_subcommand.h"

namespace Carbon {

// Options for the format subcommand.
//
// See the implementation of `Build` for documentation on members.
struct FormatOptions {
  auto Build(CommandLine::CommandBuilder& b) -> void;

  llvm::StringRef output_filename;
  llvm::SmallVector<llvm::StringRef> input_filenames;
};

// Implements the format subcommand of the driver.
class FormatSubcommand : public DriverSubcommand {
 public:
  explicit FormatSubcommand();

  auto BuildOptions(CommandLine::CommandBuilder& b) -> void override {
    options_.Build(b);
  }

  auto Run(DriverEnv& driver_env) -> DriverResult override;

 private:
  FormatOptions options_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_FORMAT_SUBCOMMAND_H_
