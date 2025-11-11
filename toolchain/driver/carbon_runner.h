// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_CARBON_RUNNER_H_
#define CARBON_TOOLCHAIN_DRIVER_CARBON_RUNNER_H_

#include <filesystem>

#include "common/error.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/driver/codegen_options.h"
#include "toolchain/driver/driver_env.h"
#include "toolchain/driver/driver_subcommand.h"
#include "toolchain/driver/tool_runner_base.h"
#include "toolchain/lower/options.h"

namespace Carbon {

// Helper class for cross-subcommand functionality.
// Allows type-safe calls of subcommands.
// TODO: May need to be adjusted once caching is refactored.
class CarbonRunner : ToolRunnerBase {
 public:
  explicit CarbonRunner(DriverEnv* driver_env);

  // TODO: Will need to be changed when compile subcommand is revised.
  auto BuildCoreLibraries(const Runtimes::Cache::Features& features,
                          Runtimes& runtimes) -> ErrorOr<std::filesystem::path>;

  auto Compile(llvm::SmallVector<std::string> input_filenames,
               Lower::OptimizationLevel opt_level, llvm::StringRef target,
               bool prelude_import) -> DriverResult;

 private:
  DriverEnv* driver_env_;
};
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_CARBON_RUNNER_H_
