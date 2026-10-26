// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_CODEGEN_OPTIONS_H_
#define CARBON_TOOLCHAIN_DRIVER_CODEGEN_OPTIONS_H_

#include <string>

#include "common/command_line.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Host.h"

namespace Carbon {

// Shared codegen-related options.
//
// See the implementation of `Build` for documentation on members.
struct CodegenOptions {
  auto Build(CommandLine::CommandBuilder& b) -> void;

  // Appends the Clang driver flags corresponding to `target_cpu`,
  // `target_cpu_tune`, and `target_cpu_features`.
  auto AppendClangArgs(llvm::SmallVectorImpl<std::string>& args) const -> void;

  // Returns the Clang driver flags corresponding to `target_cpu`,
  // `target_cpu_tune`, and `target_cpu_features`.
  auto GetClangArgs() const -> llvm::SmallVector<std::string>;

  std::string host = llvm::sys::getDefaultTargetTriple();
  llvm::StringRef target;
  llvm::StringRef target_cpu;
  llvm::StringRef target_cpu_tune;
  llvm::StringRef target_cpu_features;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_CODEGEN_OPTIONS_H_
