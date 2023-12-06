// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_INIT_LLVM_WITH_TARGETS_H_
#define CARBON_COMMON_INIT_LLVM_WITH_TARGETS_H_

#include "common/init_llvm.h"

namespace Carbon {

// A RAII class to handle initializing LLVM and shutting it down. Like
// InitLLVM, but also initializes all LLVM targets. This should only be used
// when really needed, because the LLVM targets add ~350MB of binary size to
// `-c fastbuild` binaries.
class InitLLVMWithTargets {
 public:
  // Initializes LLVM for use by a Carbon binary. On Windows, `argc` and `argv`
  // are updated to refer to properly-encoded UTF-8 versions of the command line
  // arguments.
  InitLLVMWithTargets(int& argc, char**& argv);

  // Shuts down LLVM.
  ~InitLLVMWithTargets() = default;

 private:
  Carbon::InitLLVM init_llvm;
};

}  // namespace Carbon

#endif  // CARBON_COMMON_INIT_LLVM_WITH_TARGETS_H_
