//===--- MiscTidyModule.cpp - clang-tidy ----------------------------------===//
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
#include "ArgumentCommentCheck.h"
#include "AssignOperatorSignatureCheck.h"
#include "BoolPointerImplicitConversion.h"
#include "InaccurateEraseCheck.h"
#include "InefficientAlgorithmCheck.h"
#include "SwappedArgumentsCheck.h"
#include "UndelegatedConstructor.h"
#include "UniqueptrResetRelease.h"
#include "UnusedRAII.h"
#include "UseOverride.h"

namespace clang {
namespace tidy {

class MiscModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<ArgumentCommentCheck>("misc-argument-comment");
    CheckFactories.registerCheck<AssignOperatorSignatureCheck>(
        "misc-assign-operator-signature");
    CheckFactories.registerCheck<BoolPointerImplicitConversion>(
        "misc-bool-pointer-implicit-conversion");
    CheckFactories.registerCheck<InaccurateEraseCheck>(
        "misc-inaccurate-erase");
    CheckFactories.registerCheck<InefficientAlgorithmCheck>(
        "misc-inefficient-algorithm");
    CheckFactories.registerCheck<SwappedArgumentsCheck>(
        "misc-swapped-arguments");
    CheckFactories.registerCheck<UndelegatedConstructorCheck>(
        "misc-undelegated-constructor");
    CheckFactories.registerCheck<UniqueptrResetRelease>(
        "misc-uniqueptr-reset-release");
    CheckFactories.registerCheck<UnusedRAIICheck>("misc-unused-raii");
    CheckFactories.registerCheck<UseOverride>("misc-use-override");
  }
};

// Register the MiscTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<MiscModule>
X("misc-module", "Adds miscellaneous lint checks.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the MiscModule.
volatile int MiscModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
