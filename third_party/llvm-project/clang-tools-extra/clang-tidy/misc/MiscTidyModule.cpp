//===--- MiscTidyModule.cpp - clang-tidy ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "DefinitionsInHeadersCheck.h"
#include "MisplacedConstCheck.h"
#include "NewDeleteOverloadsCheck.h"
#include "NoRecursionCheck.h"
#include "NonCopyableObjects.h"
#include "NonPrivateMemberVariablesInClassesCheck.h"
#include "RedundantExpressionCheck.h"
#include "StaticAssertCheck.h"
#include "ThrowByValueCatchByReferenceCheck.h"
#include "UnconventionalAssignOperatorCheck.h"
#include "UniqueptrResetReleaseCheck.h"
#include "UnusedAliasDeclsCheck.h"
#include "UnusedParametersCheck.h"
#include "UnusedUsingDeclsCheck.h"

namespace clang {
namespace tidy {
namespace misc {

class MiscModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<DefinitionsInHeadersCheck>(
        "misc-definitions-in-headers");
    CheckFactories.registerCheck<MisplacedConstCheck>("misc-misplaced-const");
    CheckFactories.registerCheck<NewDeleteOverloadsCheck>(
        "misc-new-delete-overloads");
    CheckFactories.registerCheck<NoRecursionCheck>("misc-no-recursion");
    CheckFactories.registerCheck<NonCopyableObjectsCheck>(
        "misc-non-copyable-objects");
    CheckFactories.registerCheck<NonPrivateMemberVariablesInClassesCheck>(
        "misc-non-private-member-variables-in-classes");
    CheckFactories.registerCheck<RedundantExpressionCheck>(
        "misc-redundant-expression");
    CheckFactories.registerCheck<StaticAssertCheck>("misc-static-assert");
    CheckFactories.registerCheck<ThrowByValueCatchByReferenceCheck>(
        "misc-throw-by-value-catch-by-reference");
    CheckFactories.registerCheck<UnconventionalAssignOperatorCheck>(
        "misc-unconventional-assign-operator");
    CheckFactories.registerCheck<UniqueptrResetReleaseCheck>(
        "misc-uniqueptr-reset-release");
    CheckFactories.registerCheck<UnusedAliasDeclsCheck>(
        "misc-unused-alias-decls");
    CheckFactories.registerCheck<UnusedParametersCheck>(
        "misc-unused-parameters");
    CheckFactories.registerCheck<UnusedUsingDeclsCheck>(
        "misc-unused-using-decls");
  }
};

} // namespace misc

// Register the MiscTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<misc::MiscModule>
    X("misc-module", "Adds miscellaneous lint checks.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the MiscModule.
volatile int MiscModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
