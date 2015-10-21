//===--- UseDefaultCheck.cpp - clang-tidy----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UseDefaultCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

static const char CtorDtor[] = "CtorDtorDecl";

void UseDefaultCheck::registerMatchers(MatchFinder *Finder) {
  if (getLangOpts().CPlusPlus) {
    Finder->addMatcher(
        cxxConstructorDecl(isDefinition(),
                           unless(hasAnyConstructorInitializer(anything())),
                           parameterCountIs(0))
            .bind(CtorDtor),
        this);
    Finder->addMatcher(cxxDestructorDecl(isDefinition()).bind(CtorDtor), this);
  }
}

void UseDefaultCheck::check(const MatchFinder::MatchResult &Result) {
  // Both CXXConstructorDecl and CXXDestructorDecl inherit from CXXMethodDecl.
  const auto *CtorDtorDecl = Result.Nodes.getNodeAs<CXXMethodDecl>(CtorDtor);

  // Discard explicitly deleted/defaulted constructors/destructors, those that
  // are not user-provided (automatically generated constructor/destructor), and
  // those with non-empty bodies.
  if (CtorDtorDecl->isDeleted() || CtorDtorDecl->isExplicitlyDefaulted() ||
      !CtorDtorDecl->isUserProvided() || !CtorDtorDecl->hasTrivialBody())
    return;

  const auto *Body = dyn_cast<CompoundStmt>(CtorDtorDecl->getBody());
  // This should never happen, since 'hasTrivialBody' checks that this is
  // actually a CompoundStmt.
  assert(Body && "Definition body is not a CompoundStmt");

  diag(CtorDtorDecl->getLocStart(),
       "use '= default' to define a trivial " +
           std::string(dyn_cast<CXXConstructorDecl>(CtorDtorDecl)
                           ? "default constructor"
                           : "destructor"))
      << FixItHint::CreateReplacement(
          CharSourceRange::getTokenRange(Body->getLBracLoc(),
                                         Body->getRBracLoc()),
          "= default;");
}

} // namespace modernize
} // namespace tidy
} // namespace clang
