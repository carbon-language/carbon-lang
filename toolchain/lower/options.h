// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_OPTIONS_H_
#define CARBON_TOOLCHAIN_LOWER_OPTIONS_H_

#include "llvm/Support/raw_ostream.h"

namespace Carbon::Lower {

enum class OptimizationLevel {
  // No optimizations beyond necessary ones like inlining always-inline
  // functions. Corresponds to Clang -O0.
  None,
  // Perform optimizations that make the build faster and don't degrade
  // debugging. Corresponds to Clang -O1 or -Og.
  Debug,
  // Optimize for binary size. Corresponds to Clang -Oz.
  Size,
  // Optimize for program execution speed. Corresponds to Clang -O3.
  Speed,
};

struct LowerToLLVMOptions {
  // Options must be set individually, not through initialization.
  explicit LowerToLLVMOptions() = default;

  // If set, enables LLVM IR verification.
  llvm::raw_ostream* llvm_verifier_stream = nullptr;

  // Whether to include debug info in lowered output.
  bool want_debug_info = false;

  // If set, enables verbose output.
  llvm::raw_ostream* vlog_stream = nullptr;

  // The optimization level to set on lowered functions by default.
  OptimizationLevel opt_level = OptimizationLevel::Debug;
};

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_OPTIONS_H_
