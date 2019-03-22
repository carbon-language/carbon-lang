//===--- OpenMPTidyModule.cpp - clang-tidy--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"

namespace clang {
namespace tidy {
namespace openmp {

/// This module is for OpenMP-specific checks.
class OpenMPModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
  }
};

// Register the OpenMPTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<OpenMPModule>
    X("openmp-module", "Adds OpenMP-specific checks.");

} // namespace openmp

// This anchor is used to force the linker to link in the generated object file
// and thus register the OpenMPModule.
volatile int OpenMPModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
