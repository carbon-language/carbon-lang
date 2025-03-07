// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/llvm_subcommand.h"

#include "common/command_line.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "toolchain/base/llvm_tools.h"
#include "toolchain/driver/llvm_runner.h"

namespace Carbon {

static constexpr CommandLine::CommandInfo SubcommandInfo = {
    .name = "llvm",
    .help = R"""(
Runs LLVM's command line tools with the provided arguments.

This subcommand provides access to a collection of LLVM's command line tools via
further subcommands. For each of these tools, their command line can be provided
using positional arguments.
)""",
};

auto LLVMOptions::Build(CommandLine::CommandBuilder& b,
                        DriverSubcommand* subcommand,
                        DriverSubcommand** selected_subcommand) -> void {
  // Add further subcommands for each LLVM tool.
  for (LLVMTool tool : LLVMTool::Tools) {
    // TODO: The subcommand info for each tool is weirdly stored in the
    // `LLVMTool` class instead of here where it makes more logical sense.
    // Either we should figure out how to move it to here or we should more
    // fully document the oddity of having it in the `LLVMTool` class.
    //
    // TODO: Currently, the command line subsystem's help isn't as user friendly
    // for the generated subcommands below this as it could be. Because each of
    // these has a completely generic and stamped out info, the `help` output is
    // very repetitive and doesn't actually contribute much information.
    b.AddSubcommand(tool.subcommand_info(), [&](auto& sub_b) {
      sub_b.AddStringPositionalArg(
          {
              .name = "ARG",
              .help = R"""(
Arguments passed to the LLVM tool.
)""",
          },
          [&](auto& arg_b) { arg_b.Append(&args); });
      sub_b.Do([=, this] {
        subcommand_tool = tool;
        *selected_subcommand = subcommand;
      });
    });
  }

  // The `llvm` subcommand can't be used directly.
  b.RequiresSubcommand();
}

LLVMSubcommand::LLVMSubcommand() : DriverSubcommand(SubcommandInfo) {}

auto LLVMSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  LLVMRunner runner(driver_env.installation, driver_env.vlog_stream);

  // Don't run arbitrary LLVM tools and libraries when fuzzing.
  if (!DisableFuzzingExternalLibraries(driver_env, "llvm")) {
    return {.success = false};
  }

  return {.success = runner.Run(options_.subcommand_tool, options_.args)};
}

}  // namespace Carbon
