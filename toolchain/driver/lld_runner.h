// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_LLD_RUNNER_H_
#define CARBON_TOOLCHAIN_DRIVER_LLD_RUNNER_H_

#include "common/ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/driver/tool_runner_base.h"
#include "toolchain/install/install_paths.h"

namespace Carbon {

// Runs LLD in a manner similar to invoking it with the provided arguments.
class LLDRunner : ToolRunnerBase {
 public:
  // Build an LLD runner that uses the provided `exe_name` and `err_stream`.
  //
  // If `verbose` is passed as true, will enable verbose logging to the
  // `err_stream` both from the runner and LLD itself.
  explicit LLDRunner(const InstallPaths* install_paths,
                     llvm::raw_ostream* vlog_stream = nullptr);

  // Run LLD as a GNU-style linker with the provided arguments.
  auto GnuLink(llvm::ArrayRef<llvm::StringRef> args) -> bool;

  // Run LLD as a Darwin-style linker with the provided arguments.
  auto DarwinLink(llvm::ArrayRef<llvm::StringRef> args) -> bool;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_LLD_RUNNER_H_
