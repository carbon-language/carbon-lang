//===--- SimplifyBooleanExpr.h clang-tidy -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_SIMPLIFY_BOOLEAN_EXPR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_SIMPLIFY_BOOLEAN_EXPR_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// \brief Looks for boolean expressions involving boolean constants and
// simplifies them to use the appropriate boolean expression directly.
///
/// Examples:
/// `if (b == true)`                           becomes `if (b)`
/// `if (b == false)`                          becomes `if (!b)`
/// `if (b && true)`                           becomes `if (b)`
/// `if (b && false)`                          becomes `if (false)`
/// `if (b || true)`                           becomes `if (true)`
/// `if (b || false)`                          becomes `if (b)`
/// `e ? true : false`                         becomes `e`
/// `e ? false : true`                         becomes `!e`
/// `if (true) t(); else f();`                 becomes `t();`
/// `if (false) t(); else f();`                becomes `f();`
/// `if (e) return true; else return false;`   becomes `return (e);`
/// `if (e) return false; else return true;`   becomes `return !(e);`
/// `if (e) b = true; else b = false;`         becomes `b = e;`
/// `if (e) b = false; else b = true;`         becomes `b = !(e);`
///
class SimplifyBooleanExprCheck : public ClangTidyCheck {
public:
  SimplifyBooleanExprCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void matchBoolBinOpExpr(ast_matchers::MatchFinder *Finder, bool Value,
                          StringRef OperatorName, StringRef BooleanId);

  void matchExprBinOpBool(ast_matchers::MatchFinder *Finder, bool Value,
                          StringRef OperatorName, StringRef BooleanId);

  void matchBoolCompOpExpr(ast_matchers::MatchFinder *Finder, bool Value,
                           StringRef OperatorName, StringRef BooleanId);

  void matchExprCompOpBool(ast_matchers::MatchFinder *Finder, bool Value,
                           StringRef OperatorName, StringRef BooleanId);

  void matchBoolCondition(ast_matchers::MatchFinder *Finder, bool Value,
                          StringRef BooleanId);

  void matchTernaryResult(ast_matchers::MatchFinder *Finder, bool Value,
                          StringRef TernaryId);

  void matchIfReturnsBool(ast_matchers::MatchFinder *Finder, bool Value,
                          StringRef Id);

  void matchIfAssignsBool(ast_matchers::MatchFinder *Finder, bool Value,
                          StringRef Id);

  void
  replaceWithExpression(const ast_matchers::MatchFinder::MatchResult &Result,
                        const CXXBoolLiteralExpr *BoolLiteral, bool UseLHS,
                        bool Negated = false);

  void
  replaceWithThenStatement(const ast_matchers::MatchFinder::MatchResult &Result,
                           const CXXBoolLiteralExpr *BoolLiteral);

  void
  replaceWithElseStatement(const ast_matchers::MatchFinder::MatchResult &Result,
                           const CXXBoolLiteralExpr *FalseConditionRemoved);

  void
  replaceWithCondition(const ast_matchers::MatchFinder::MatchResult &Result,
                       const ConditionalOperator *Ternary,
                       bool Negated = false);

  void replaceWithReturnCondition(
      const ast_matchers::MatchFinder::MatchResult &Result, const IfStmt *If,
      bool Negated = false);

  void
  replaceWithAssignment(const ast_matchers::MatchFinder::MatchResult &Result,
                        const IfStmt *If, bool Negated = false);
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_SIMPLIFY_BOOLEAN_EXPR_H
