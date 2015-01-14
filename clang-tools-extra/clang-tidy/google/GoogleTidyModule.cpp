//===--- GoogleTidyModule.cpp - clang-tidy --------------------------------===//
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
#include "../readability/BracesAroundStatementsCheck.h"
#include "../readability/FunctionSize.h"
#include "../readability/NamespaceCommentCheck.h"
#include "../readability/RedundantSmartptrGet.h"
#include "AvoidCStyleCastsCheck.h"
#include "ExplicitConstructorCheck.h"
#include "ExplicitMakePairCheck.h"
#include "IntegerTypesCheck.h"
#include "MemsetZeroLengthCheck.h"
#include "NamedParameterCheck.h"
#include "OverloadedUnaryAndCheck.h"
#include "StringReferenceMemberCheck.h"
#include "TodoCommentCheck.h"
#include "UnnamedNamespaceInHeaderCheck.h"
#include "UsingNamespaceDirectiveCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

class GoogleModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<build::ExplicitMakePairCheck>(
        "google-build-explicit-make-pair");
    CheckFactories.registerCheck<build::UnnamedNamespaceInHeaderCheck>(
        "google-build-namespaces");
    CheckFactories.registerCheck<build::UsingNamespaceDirectiveCheck>(
        "google-build-using-namespace");
    CheckFactories.registerCheck<ExplicitConstructorCheck>(
        "google-explicit-constructor");
    CheckFactories.registerCheck<runtime::IntegerTypesCheck>(
        "google-runtime-int");
    CheckFactories.registerCheck<runtime::OverloadedUnaryAndCheck>(
        "google-runtime-operator");
    CheckFactories.registerCheck<runtime::StringReferenceMemberCheck>(
        "google-runtime-member-string-references");
    CheckFactories.registerCheck<runtime::MemsetZeroLengthCheck>(
        "google-runtime-memset");
    CheckFactories.registerCheck<readability::AvoidCStyleCastsCheck>(
        "google-readability-casting");
    CheckFactories.registerCheck<readability::NamedParameterCheck>(
        "google-readability-function");
    CheckFactories.registerCheck<readability::TodoCommentCheck>(
        "google-readability-todo");
    CheckFactories.registerCheck<readability::BracesAroundStatementsCheck>(
        "google-readability-braces-around-statements");
    CheckFactories.registerCheck<readability::FunctionSizeCheck>(
        "google-readability-function-size");
    CheckFactories.registerCheck<readability::NamespaceCommentCheck>(
        "google-readability-namespace-comments");
    CheckFactories.registerCheck<readability::RedundantSmartptrGet>(
        "google-readability-redundant-smartptr-get");
  }

  ClangTidyOptions getModuleOptions() override {
    ClangTidyOptions Options;
    auto &Opts = Options.CheckOptions;
    Opts["google-readability-braces-around-statements.ShortStatementLines"] =
        "1";
    Opts["google-readability-function-size.StatementThreshold"] = "800";
    Opts["google-readability-namespace-comments.ShortNamespaceLines"] = "10";
    Opts["google-readability-namespace-comments.SpacesBeforeComments"] = "2";
    return Options;
  }
};

// Register the GoogleTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<GoogleModule> X("google-module",
                                                    "Adds Google lint checks.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the GoogleModule.
volatile int GoogleModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
