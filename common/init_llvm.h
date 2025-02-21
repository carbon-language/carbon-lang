// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_INIT_LLVM_H_
#define CARBON_COMMON_INIT_LLVM_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/InitLLVM.h"

namespace Carbon {

// A RAII class to handle initializing LLVM and shutting it down. An instance of
// this class should be created in the `main` function of each Carbon binary
// that interacts with LLVM, before `argc` and `argv` are first inspected.
class InitLLVM {
 public:
  // Initializes LLVM for use by a Carbon binary. On Windows, `argc` and `argv`
  // are updated to refer to properly-encoded UTF-8 versions of the command line
  // arguments.
  explicit InitLLVM(int& argc, char**& argv);

  // Shuts down LLVM.
  ~InitLLVM() = default;

 private:
  using InitializeTargetsFn = auto() -> void;

  llvm::InitLLVM init_llvm_;
  llvm::SmallVector<char*> args_;

  // A pointer to the LLVM target initialization function, if :all_llvm_targets
  // is linked in. Otherwise nullptr.
  // NOLINTNEXTLINE(readability-identifier-naming): Constant after static init.
  static InitializeTargetsFn* InitializeTargets;

  // The initializer of this static data member populates `InitializeTargets`.
  // Defined only if :all_llvm_targets is linked in. This is a member so that
  // it has access to `InitializeTargets`.
  static const char RegisterTargets;
};

}  // namespace Carbon

#endif  // CARBON_COMMON_INIT_LLVM_H_
