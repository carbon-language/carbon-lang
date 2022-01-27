//===--- SetLongJmpCheck.cpp - clang-tidy----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SetLongJmpCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cert {

namespace {
const char DiagWording[] =
    "do not call %0; consider using exception handling instead";

class SetJmpMacroCallbacks : public PPCallbacks {
  SetLongJmpCheck &Check;

public:
  explicit SetJmpMacroCallbacks(SetLongJmpCheck &Check) : Check(Check) {}

  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override {
    const auto *II = MacroNameTok.getIdentifierInfo();
    if (!II)
      return;

    if (II->getName() == "setjmp")
      Check.diag(Range.getBegin(), DiagWording) << II;
  }
};
} // namespace

void SetLongJmpCheck::registerPPCallbacks(const SourceManager &SM,
                                          Preprocessor *PP,
                                          Preprocessor *ModuleExpanderPP) {
  // Per [headers]p5, setjmp must be exposed as a macro instead of a function,
  // despite the allowance in C for setjmp to also be an extern function.
  PP->addPPCallbacks(std::make_unique<SetJmpMacroCallbacks>(*this));
}

void SetLongJmpCheck::registerMatchers(MatchFinder *Finder) {
  // In case there is an implementation that happens to define setjmp as a
  // function instead of a macro, this will also catch use of it. However, we
  // are primarily searching for uses of longjmp.
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasAnyName("setjmp", "longjmp"))))
          .bind("expr"),
      this);
}

void SetLongJmpCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *E = Result.Nodes.getNodeAs<CallExpr>("expr");
  diag(E->getExprLoc(), DiagWording) << cast<NamedDecl>(E->getCalleeDecl());
}

} // namespace cert
} // namespace tidy
} // namespace clang
