// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_COMPILE_SUBCOMMAND_H_
#define CARBON_TOOLCHAIN_DRIVER_COMPILE_SUBCOMMAND_H_

#include "common/command_line.h"
#include "toolchain/driver/compile_options.h"
#include "toolchain/driver/driver_env.h"
#include "toolchain/driver/driver_subcommand.h"

namespace Carbon {

// Options for the compile subcommand.
struct CompileSubcommandOptions {
  auto Build(CommandLine::CommandBuilder& b) -> void {
    compile_options.BuildForCompileSubcommand(b, &codegen_options);
  }

  CodegenOptions codegen_options;
  CompileOptions compile_options;
};

// Implements the compile subcommand of the driver.
class CompileSubcommand : public DriverSubcommand {
 public:
  explicit CompileSubcommand();

  auto BuildOptions(CommandLine::CommandBuilder& b) -> void override {
    options_.Build(b);
  }

  auto Run(DriverEnv& driver_env) -> DriverResult override;

 private:
  CompileSubcommandOptions options_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_COMPILE_SUBCOMMAND_H_
