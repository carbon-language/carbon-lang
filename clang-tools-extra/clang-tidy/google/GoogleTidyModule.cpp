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
#include "OverloadedUnaryAndCheck.h"

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
        "google-explicit-constructor",
        new ClangTidyCheckFactory<ExplicitConstructorCheck>());
    CheckFactories.addCheckFactory(
        "google-runtime-operator",
        new ClangTidyCheckFactory<runtime::OverloadedUnaryAndCheck>());
    CheckFactories.addCheckFactory(
        "google-readability-casting",
        new ClangTidyCheckFactory<readability::AvoidCStyleCastsCheck>());
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
