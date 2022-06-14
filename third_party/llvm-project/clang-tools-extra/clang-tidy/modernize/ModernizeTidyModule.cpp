//===--- ModernizeTidyModule.cpp - clang-tidy -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "AvoidBindCheck.h"
#include "AvoidCArraysCheck.h"
#include "ConcatNestedNamespacesCheck.h"
#include "DeprecatedHeadersCheck.h"
#include "DeprecatedIosBaseAliasesCheck.h"
#include "LoopConvertCheck.h"
#include "MacroToEnumCheck.h"
#include "MakeSharedCheck.h"
#include "MakeUniqueCheck.h"
#include "PassByValueCheck.h"
#include "RawStringLiteralCheck.h"
#include "RedundantVoidArgCheck.h"
#include "ReplaceAutoPtrCheck.h"
#include "ReplaceDisallowCopyAndAssignMacroCheck.h"
#include "ReplaceRandomShuffleCheck.h"
#include "ReturnBracedInitListCheck.h"
#include "ShrinkToFitCheck.h"
#include "UnaryStaticAssertCheck.h"
#include "UseAutoCheck.h"
#include "UseBoolLiteralsCheck.h"
#include "UseDefaultMemberInitCheck.h"
#include "UseEmplaceCheck.h"
#include "UseEqualsDefaultCheck.h"
#include "UseEqualsDeleteCheck.h"
#include "UseNodiscardCheck.h"
#include "UseNoexceptCheck.h"
#include "UseNullptrCheck.h"
#include "UseOverrideCheck.h"
#include "UseTrailingReturnTypeCheck.h"
#include "UseTransparentFunctorsCheck.h"
#include "UseUncaughtExceptionsCheck.h"
#include "UseUsingCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

class ModernizeModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<AvoidBindCheck>("modernize-avoid-bind");
    CheckFactories.registerCheck<AvoidCArraysCheck>("modernize-avoid-c-arrays");
    CheckFactories.registerCheck<ConcatNestedNamespacesCheck>(
        "modernize-concat-nested-namespaces");
    CheckFactories.registerCheck<DeprecatedHeadersCheck>(
        "modernize-deprecated-headers");
    CheckFactories.registerCheck<DeprecatedIosBaseAliasesCheck>(
        "modernize-deprecated-ios-base-aliases");
    CheckFactories.registerCheck<LoopConvertCheck>("modernize-loop-convert");
    CheckFactories.registerCheck<MacroToEnumCheck>("modernize-macro-to-enum");
    CheckFactories.registerCheck<MakeSharedCheck>("modernize-make-shared");
    CheckFactories.registerCheck<MakeUniqueCheck>("modernize-make-unique");
    CheckFactories.registerCheck<PassByValueCheck>("modernize-pass-by-value");
    CheckFactories.registerCheck<RawStringLiteralCheck>(
        "modernize-raw-string-literal");
    CheckFactories.registerCheck<RedundantVoidArgCheck>(
        "modernize-redundant-void-arg");
    CheckFactories.registerCheck<ReplaceAutoPtrCheck>(
        "modernize-replace-auto-ptr");
    CheckFactories.registerCheck<ReplaceDisallowCopyAndAssignMacroCheck>(
        "modernize-replace-disallow-copy-and-assign-macro");
    CheckFactories.registerCheck<ReplaceRandomShuffleCheck>(
        "modernize-replace-random-shuffle");
    CheckFactories.registerCheck<ReturnBracedInitListCheck>(
        "modernize-return-braced-init-list");
    CheckFactories.registerCheck<ShrinkToFitCheck>("modernize-shrink-to-fit");
    CheckFactories.registerCheck<UnaryStaticAssertCheck>(
        "modernize-unary-static-assert");
    CheckFactories.registerCheck<UseAutoCheck>("modernize-use-auto");
    CheckFactories.registerCheck<UseBoolLiteralsCheck>(
        "modernize-use-bool-literals");
    CheckFactories.registerCheck<UseDefaultMemberInitCheck>(
        "modernize-use-default-member-init");
    CheckFactories.registerCheck<UseEmplaceCheck>("modernize-use-emplace");
    CheckFactories.registerCheck<UseEqualsDefaultCheck>("modernize-use-equals-default");
    CheckFactories.registerCheck<UseEqualsDeleteCheck>(
        "modernize-use-equals-delete");
    CheckFactories.registerCheck<UseNodiscardCheck>(
        "modernize-use-nodiscard");
    CheckFactories.registerCheck<UseNoexceptCheck>("modernize-use-noexcept");
    CheckFactories.registerCheck<UseNullptrCheck>("modernize-use-nullptr");
    CheckFactories.registerCheck<UseOverrideCheck>("modernize-use-override");
    CheckFactories.registerCheck<UseTrailingReturnTypeCheck>(
        "modernize-use-trailing-return-type");
    CheckFactories.registerCheck<UseTransparentFunctorsCheck>(
        "modernize-use-transparent-functors");
    CheckFactories.registerCheck<UseUncaughtExceptionsCheck>(
        "modernize-use-uncaught-exceptions");
    CheckFactories.registerCheck<UseUsingCheck>("modernize-use-using");
  }

  ClangTidyOptions getModuleOptions() override {
    ClangTidyOptions Options;
    auto &Opts = Options.CheckOptions;
    // For types whose size in bytes is above this threshold, we prefer taking a
    // const-reference than making a copy.
    Opts["modernize-loop-convert.MaxCopySize"] = "16";

    Opts["modernize-loop-convert.MinConfidence"] = "reasonable";
    Opts["modernize-loop-convert.NamingStyle"] = "CamelCase";
    Opts["modernize-pass-by-value.IncludeStyle"] = "llvm";    // Also: "google".
    Opts["modernize-replace-auto-ptr.IncludeStyle"] = "llvm"; // Also: "google".

    // Comma-separated list of macros that behave like NULL.
    Opts["modernize-use-nullptr.NullMacros"] = "NULL";
    return Options;
  }
};

// Register the ModernizeTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<ModernizeModule> X("modernize-module",
                                                       "Add modernize checks.");

} // namespace modernize

// This anchor is used to force the linker to link in the generated object file
// and thus register the ModernizeModule.
volatile int ModernizeModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
