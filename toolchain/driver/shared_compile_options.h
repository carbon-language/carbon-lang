// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_SHARED_COMPILE_OPTIONS_H_
#define CARBON_TOOLCHAIN_DRIVER_SHARED_COMPILE_OPTIONS_H_

#include <memory>

#include "common/command_line.h"
#include "common/error.h"
#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "toolchain/check/check.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/driver/codegen_options.h"
#include "toolchain/driver/driver_env.h"
#include "toolchain/lower/lower.h"

namespace Carbon {

// Options for Carbon compilation. This struct is shared between the
// `build` and `compile` subcommands, supporting different flags
// for each subcomand.
//
// Note that the `Build()` function provides only the common flags shared
// between both subcommands, and each subcommand provides their own specialized
// flags in their respective `Options` structs.
//
// Members are documented in the `Build` function.
struct SharedCompileOptions {
  auto Build(CommandLine::CommandBuilder& b) -> void;

  // Validate the target before passing to clang.
  auto ValidateTarget(Diagnostics::NoLocEmitter& emitter)
      -> ErrorOr<const llvm::Target*>;

  // Build a clang invocation. We do this regardless of whether we're running
  // check, because this is essentially performing further option validation,
  // and we generally validate all options even if we're not using them for the
  // selected phases of compilation. We also use Clang's target option handling
  // to configure our target, to ensure that we are using the same ABI for both
  // the C++ and Carbon parts of the compilation.
  // TODO: Share any arguments we specify here with the `carbon clang`
  // subcommand.
  auto BuildClangInvocation(DriverEnv& driver_env)
      -> ErrorOr<std::shared_ptr<clang::CompilerInvocation>>;

  Lower::OptimizationLevel opt_level = Lower::OptimizationLevel::Debug;
  CodegenOptions codegen_options;

  llvm::SmallVector<llvm::StringRef> input_filenames;
  llvm::SmallVector<llvm::StringRef> clang_args;

  bool include_debug_info = true;
  bool run_llvm_verifier = true;

  static auto GetLLVMOptimizationLevel(Lower::OptimizationLevel opt_level)
      -> llvm::OptimizationLevel;

  // Get the `-O` flag corresponding to an optimization level.
  static auto GetClangOptimizationFlag(Lower::OptimizationLevel opt_level)
      -> llvm::StringLiteral;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_SHARED_COMPILE_OPTIONS_H_
