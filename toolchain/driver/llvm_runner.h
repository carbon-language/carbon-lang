// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_LLVM_RUNNER_H_
#define CARBON_TOOLCHAIN_DRIVER_LLVM_RUNNER_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/base/llvm_tools.h"
#include "toolchain/driver/tool_runner_base.h"

namespace Carbon {

// Runs any of the LLVM tools in a manner similar to invoking it with the
// provided arguments.
class LLVMRunner : ToolRunnerBase {
 public:
  using ToolRunnerBase::ToolRunnerBase;

  auto Run(LLVMTool tool, llvm::ArrayRef<llvm::StringRef> args) -> bool;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_LLVM_RUNNER_H_
