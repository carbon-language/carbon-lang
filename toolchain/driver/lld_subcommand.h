// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_LLD_SUBCOMMAND_H_
#define CARBON_TOOLCHAIN_DRIVER_LLD_SUBCOMMAND_H_

#include "common/command_line.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/driver/driver_env.h"
#include "toolchain/driver/driver_subcommand.h"

namespace Carbon {

// Options for the LLD subcommand, which is just a thin wrapper.
//
// See the implementation of `Build` for documentation on members.
struct LLDOptions {
  enum class Platform {
    Gnu,
    Darwin,
  };

  auto Build(CommandLine::CommandBuilder& b) -> void;

  Platform platform;
  llvm::SmallVector<llvm::StringRef> args;
};

// Implements the LLD subcommand of the driver.
class LLDSubcommand : public DriverSubcommand {
 public:
  explicit LLDSubcommand();

  auto BuildOptions(CommandLine::CommandBuilder& b) -> void override {
    options_.Build(b);
  }

  auto Run(DriverEnv& driver_env) -> DriverResult override;

 private:
  LLDOptions options_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_LLD_SUBCOMMAND_H_
