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
    CheckFactories.addCheckFactory(
        "google-build-explicit-make-pair",
        new ClangTidyCheckFactory<build::ExplicitMakePairCheck>());
    CheckFactories.addCheckFactory(
        "google-build-namespaces",
        new ClangTidyCheckFactory<build::UnnamedNamespaceInHeaderCheck>());
    CheckFactories.addCheckFactory(
        "google-build-using-namespace",
        new ClangTidyCheckFactory<build::UsingNamespaceDirectiveCheck>());
    CheckFactories.addCheckFactory(
        "google-explicit-constructor",
        new ClangTidyCheckFactory<ExplicitConstructorCheck>());
    CheckFactories.addCheckFactory(
        "google-runtime-int",
        new ClangTidyCheckFactory<runtime::IntegerTypesCheck>());
    CheckFactories.addCheckFactory(
        "google-runtime-operator",
        new ClangTidyCheckFactory<runtime::OverloadedUnaryAndCheck>());
    CheckFactories.addCheckFactory(
        "google-runtime-member-string-references",
        new ClangTidyCheckFactory<runtime::StringReferenceMemberCheck>());
    CheckFactories.addCheckFactory(
        "google-runtime-memset",
        new ClangTidyCheckFactory<runtime::MemsetZeroLengthCheck>());
    CheckFactories.addCheckFactory(
        "google-readability-casting",
        new ClangTidyCheckFactory<readability::AvoidCStyleCastsCheck>());
    CheckFactories.addCheckFactory(
        "google-readability-function",
        new ClangTidyCheckFactory<readability::NamedParameterCheck>());
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
