// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_LLD_RUNNER_H_
#define CARBON_TOOLCHAIN_DRIVER_LLD_RUNNER_H_

#include "common/ostream.h"
#include "lld/Common/Driver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/driver/tool_runner_base.h"
#include "toolchain/install/install_paths.h"

namespace Carbon {

// Runs LLD in a manner similar to invoking it with the provided arguments.
class LldRunner : ToolRunnerBase {
 public:
  using ToolRunnerBase::ToolRunnerBase;

  // Run LLD as a GNU-style linker with the provided arguments.
  auto ElfLink(llvm::ArrayRef<llvm::StringRef> args) -> bool;

  // Run LLD as a Darwin-style linker with the provided arguments.
  auto MachOLink(llvm::ArrayRef<llvm::StringRef> args) -> bool;

 private:
  auto LinkHelper(llvm::StringLiteral label,
                  llvm::ArrayRef<llvm::StringRef> args, const std::string& path,
                  lld::DriverDef driver_def) -> bool;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_LLD_RUNNER_H_
