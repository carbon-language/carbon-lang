//===--- ReadabilityTidyModule.cpp - clang-tidy ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "BracesAroundStatementsCheck.h"
#include "ContainerSizeEmpty.h"
#include "ElseAfterReturnCheck.h"
#include "FunctionSize.h"
#include "RedundantSmartptrGet.h"
#include "ShrinkToFitCheck.h"

namespace clang {
namespace tidy {
namespace readability {

class ReadabilityModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<BracesAroundStatementsCheck>(
        "readability-braces-around-statements");
    CheckFactories.registerCheck<ContainerSizeEmptyCheck>(
        "readability-container-size-empty");
    CheckFactories.registerCheck<ElseAfterReturnCheck>(
        "readability-else-after-return");
    CheckFactories.registerCheck<FunctionSizeCheck>(
        "readability-function-size");
    CheckFactories.registerCheck<RedundantSmartptrGet>(
        "readability-redundant-smartptr-get");
    CheckFactories.registerCheck<ShrinkToFitCheck>(
        "readability-shrink-to-fit");
  }
};

} // namespace readability

// Register the MiscTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<readability::ReadabilityModule>
X("readability-module", "Adds readability-related checks.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the MiscModule.
volatile int ReadabilityModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
