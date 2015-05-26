//===--- NoexceptMoveCtorsCheck.cpp - clang-tidy---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NoexceptMoveCtorsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

void NoexceptMoveCtorsCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      methodDecl(anyOf(constructorDecl(), hasOverloadedOperatorName("=")),
                 unless(isImplicit()), unless(isDeleted()))
          .bind("decl"),
      this);
}

void NoexceptMoveCtorsCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Decl = Result.Nodes.getNodeAs<CXXMethodDecl>("decl")) {
    StringRef MethodType = "assignment operator";
    if (const auto *Ctor = dyn_cast<CXXConstructorDecl>(Decl)) {
      if (!Ctor->isMoveConstructor())
        return;
      MethodType = "constructor";
    } else if (!Decl->isMoveAssignmentOperator()) {
      return;
    }

    const auto *ProtoType = Decl->getType()->getAs<FunctionProtoType>();
    switch(ProtoType->getNoexceptSpec(*Result.Context)) {
      case  FunctionProtoType::NR_NoNoexcept:
        diag(Decl->getLocation(), "move %0s should be marked noexcept")
            << MethodType;
        // FIXME: Add a fixit.
        break;
      case FunctionProtoType::NR_Throw:
        // Don't complain about nothrow(false), but complain on nothrow(expr)
        // where expr evaluates to false.
        if (const Expr *E = ProtoType->getNoexceptExpr()) {
          if (isa<CXXBoolLiteralExpr>(E))
            break;
          diag(E->getExprLoc(),
               "noexcept specifier on the move %0 evaluates to 'false'")
              << MethodType;
        }
        break;
      case FunctionProtoType::NR_Nothrow:
      case FunctionProtoType::NR_Dependent:
      case FunctionProtoType::NR_BadNoexcept:
        break;
    }
  }
}

} // namespace tidy
} // namespace clang

