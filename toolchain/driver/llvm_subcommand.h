// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_LLVM_SUBCOMMAND_H_
#define CARBON_TOOLCHAIN_DRIVER_LLVM_SUBCOMMAND_H_

#include "common/command_line.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/base/llvm_tools.h"
#include "toolchain/driver/driver_env.h"
#include "toolchain/driver/driver_subcommand.h"

namespace Carbon {

// Options for the LLVM subcommand.
//
// See the implementation of `Build` for documentation on members.
struct LLVMOptions {
  // Build the LLVM subcommand options using `b`.
  //
  // When this top-level subcommand is selected (potentially through a nested
  // sub-subcommand), the `selected_subcommand` will be set to point to
  // `subcommand` to reflect that.
  auto Build(CommandLine::CommandBuilder& b, DriverSubcommand* subcommand,
             DriverSubcommand** selected_subcommand) -> void;

  LLVMTool subcommand_tool;
  llvm::SmallVector<llvm::StringRef> args;
};

// Implement the LLVM subcommand of the driver.
//
// This provides access to the full collection of LLVM command line tools.
class LLVMSubcommand : public DriverSubcommand {
 public:
  explicit LLVMSubcommand();

  // The LLVM subcommand uses a custom subcommand structure, so `BuildOptions`
  // is a no-op and we override the more complex layer.
  auto BuildOptions(CommandLine::CommandBuilder& /*b*/) -> void override {
    CARBON_FATAL("Unused.");
  }
  auto BuildOptionsAndSetAction(CommandLine::CommandBuilder& b,
                                DriverSubcommand** selected_subcommand)
      -> void override {
    options_.Build(b, this, selected_subcommand);
  }

  auto Run(DriverEnv& driver_env) -> DriverResult override;

 private:
  LLVMOptions options_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_LLVM_SUBCOMMAND_H_
