//===--- GoogleTidyModule.cpp - clang-tidy --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "GoogleTidyModule.h"
#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PPCallbacks.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

void
ExplicitConstructorCheck::registerMatchers(ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(constructorDecl().bind("construct"), this);
}

void ExplicitConstructorCheck::check(const MatchFinder::MatchResult &Result) {
  const CXXConstructorDecl *Ctor =
      Result.Nodes.getNodeAs<CXXConstructorDecl>("construct");
  if (!Ctor->isExplicit() && !Ctor->isImplicit() && Ctor->getNumParams() >= 1 &&
      Ctor->getMinRequiredArguments() <= 1) {
    SourceLocation Loc = Ctor->getLocation();
    Context->Diag(Loc, "Single-argument constructors must be explicit")
        << FixItHint::CreateInsertion(Loc, "explicit ");
  }
}

class GoogleModule : public ClangTidyModule {
public:
  virtual void addCheckFactories(ClangTidyCheckFactories &CheckFactories) {
    CheckFactories.addCheckFactory(
        "google-explicit-constructor",
        new ClangTidyCheckFactory<ExplicitConstructorCheck>());
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
