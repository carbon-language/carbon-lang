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
#include "AvoidCStyleCastsCheck.h"
#include "ExplicitConstructorCheck.h"
#include "ExplicitMakePairCheck.h"
#include "IntegerTypesCheck.h"
#include "MemsetZeroLengthCheck.h"
#include "NamedParameterCheck.h"
#include "OverloadedUnaryAndCheck.h"
#include "StringReferenceMemberCheck.h"
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
