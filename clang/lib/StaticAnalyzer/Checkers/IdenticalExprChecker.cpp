//== IdenticalExprChecker.cpp - Identical expression checker----------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This defines IdenticalExprChecker, a check that warns about
/// unintended use of identical expressions.
///
/// It checks for use of identical expressions with comparison operators and
/// inside conditional expressions.
///
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

static bool isIdenticalExpr(const ASTContext &Ctx, const Expr *Expr1,
                            const Expr *Expr2, bool IgnoreSideEffects = false);
//===----------------------------------------------------------------------===//
// FindIdenticalExprVisitor - Identify nodes using identical expressions.
//===----------------------------------------------------------------------===//

namespace {
class FindIdenticalExprVisitor
    : public RecursiveASTVisitor<FindIdenticalExprVisitor> {
public:
  explicit FindIdenticalExprVisitor(BugReporter &B,
                                    const CheckerBase *Checker,
                                    AnalysisDeclContext *A)
      : BR(B), Checker(Checker), AC(A) {}
  // FindIdenticalExprVisitor only visits nodes
  // that are binary operators or conditional operators.
  bool VisitBinaryOperator(const BinaryOperator *B);
  bool VisitConditionalOperator(const ConditionalOperator *C);

private:
  BugReporter &BR;
  const CheckerBase *Checker;
  AnalysisDeclContext *AC;
};
} // end anonymous namespace

bool FindIdenticalExprVisitor::VisitBinaryOperator(const BinaryOperator *B) {
  BinaryOperator::Opcode Op = B->getOpcode();
  if (!BinaryOperator::isComparisonOp(Op))
    return true;
  //
  // Special case for floating-point representation.
  //
  // If expressions on both sides of comparison operator are of type float,
  // then for some comparison operators no warning shall be
  // reported even if the expressions are identical from a symbolic point of
  // view. Comparison between expressions, declared variables and literals
  // are treated differently.
  //
  // != and == between float literals that have the same value should NOT warn.
  // < > between float literals that have the same value SHOULD warn.
  //
  // != and == between the same float declaration should NOT warn.
  // < > between the same float declaration SHOULD warn.
  //
  // != and == between eq. expressions that evaluates into float
  //           should NOT warn.
  // < >       between eq. expressions that evaluates into float
  //           should NOT warn.
  //
  const Expr *LHS = B->getLHS()->IgnoreParenImpCasts();
  const Expr *RHS = B->getRHS()->IgnoreParenImpCasts();

  const DeclRefExpr *DeclRef1 = dyn_cast<DeclRefExpr>(LHS);
  const DeclRefExpr *DeclRef2 = dyn_cast<DeclRefExpr>(RHS);
  const FloatingLiteral *FloatLit1 = dyn_cast<FloatingLiteral>(LHS);
  const FloatingLiteral *FloatLit2 = dyn_cast<FloatingLiteral>(RHS);
  if ((DeclRef1) && (DeclRef2)) {
    if ((DeclRef1->getType()->hasFloatingRepresentation()) &&
        (DeclRef2->getType()->hasFloatingRepresentation())) {
      if (DeclRef1->getDecl() == DeclRef2->getDecl()) {
        if ((Op == BO_EQ) || (Op == BO_NE)) {
          return true;
        }
      }
    }
  } else if ((FloatLit1) && (FloatLit2)) {
    if (FloatLit1->getValue().bitwiseIsEqual(FloatLit2->getValue())) {
      if ((Op == BO_EQ) || (Op == BO_NE)) {
        return true;
      }
    }
  } else if (LHS->getType()->hasFloatingRepresentation()) {
    // If any side of comparison operator still has floating-point
    // representation, then it's an expression. Don't warn.
    // Here only LHS is checked since RHS will be implicit casted to float.
    return true;
  } else {
    // No special case with floating-point representation, report as usual.
  }

  if (isIdenticalExpr(AC->getASTContext(), B->getLHS(), B->getRHS())) {
    PathDiagnosticLocation ELoc =
        PathDiagnosticLocation::createOperatorLoc(B, BR.getSourceManager());
    StringRef Message;
    if (((Op == BO_EQ) || (Op == BO_LE) || (Op == BO_GE)))
      Message = "comparison of identical expressions always evaluates to true";
    else
      Message = "comparison of identical expressions always evaluates to false";
    BR.EmitBasicReport(AC->getDecl(), Checker,
                       "Compare of identical expressions",
                       categories::LogicError, Message, ELoc);
  }
  // We want to visit ALL nodes (subexpressions of binary comparison
  // expressions too) that contains comparison operators.
  // True is always returned to traverse ALL nodes.
  return true;
}

bool FindIdenticalExprVisitor::VisitConditionalOperator(
    const ConditionalOperator *C) {

  // Check if expressions in conditional expression are identical
  // from a symbolic point of view.

  if (isIdenticalExpr(AC->getASTContext(), C->getTrueExpr(),
                      C->getFalseExpr(), true)) {
    PathDiagnosticLocation ELoc =
        PathDiagnosticLocation::createConditionalColonLoc(
            C, BR.getSourceManager());

    SourceRange Sr[2];
    Sr[0] = C->getTrueExpr()->getSourceRange();
    Sr[1] = C->getFalseExpr()->getSourceRange();
    BR.EmitBasicReport(
        AC->getDecl(), Checker,
        "Identical expressions in conditional expression",
        categories::LogicError,
        "identical expressions on both sides of ':' in conditional expression",
        ELoc, Sr);
  }
  // We want to visit ALL nodes (expressions in conditional
  // expressions too) that contains conditional operators,
  // thus always return true to traverse ALL nodes.
  return true;
}

/// \brief Determines whether two expression trees are identical regarding
/// operators and symbols.
///
/// Exceptions: expressions containing macros or functions with possible side
/// effects are never considered identical.
/// Limitations: (t + u) and (u + t) are not considered identical.
/// t*(u + t) and t*u + t*t are not considered identical.
///
static bool isIdenticalExpr(const ASTContext &Ctx, const Expr *Expr1,
                            const Expr *Expr2, bool IgnoreSideEffects) {
  // If Expr1 & Expr2 are of different class then they are not
  // identical expression.
  if (Expr1->getStmtClass() != Expr2->getStmtClass())
    return false;
  // If Expr1 has side effects then don't warn even if expressions
  // are identical.
  if (!IgnoreSideEffects && Expr1->HasSideEffects(Ctx))
    return false;
  // If either expression comes from a macro then don't warn even if
  // the expressions are identical.
  if ((Expr1->getExprLoc().isMacroID()) || (Expr2->getExprLoc().isMacroID()))
    return false;
  // If all children of two expressions are identical, return true.
  Expr::const_child_iterator I1 = Expr1->child_begin();
  Expr::const_child_iterator I2 = Expr2->child_begin();
  while (I1 != Expr1->child_end() && I2 != Expr2->child_end()) {
    const Expr *Child1 = dyn_cast<Expr>(*I1);
    const Expr *Child2 = dyn_cast<Expr>(*I2);
    if (!Child1 || !Child2 || !isIdenticalExpr(Ctx, Child1, Child2,
                                               IgnoreSideEffects))
      return false;
    ++I1;
    ++I2;
  }
  // If there are different number of children in the expressions, return false.
  // (TODO: check if this is a redundant condition.)
  if (I1 != Expr1->child_end())
    return false;
  if (I2 != Expr2->child_end())
    return false;

  switch (Expr1->getStmtClass()) {
  default:
    return false;
  case Stmt::CallExprClass:
  case Stmt::ArraySubscriptExprClass:
  case Stmt::CStyleCastExprClass:
  case Stmt::ImplicitCastExprClass:
  case Stmt::ParenExprClass:
    return true;
  case Stmt::BinaryOperatorClass: {
    const BinaryOperator *BinOp1 = cast<BinaryOperator>(Expr1);
    const BinaryOperator *BinOp2 = cast<BinaryOperator>(Expr2);
    return BinOp1->getOpcode() == BinOp2->getOpcode();
  }
  case Stmt::CharacterLiteralClass: {
    const CharacterLiteral *CharLit1 = cast<CharacterLiteral>(Expr1);
    const CharacterLiteral *CharLit2 = cast<CharacterLiteral>(Expr2);
    return CharLit1->getValue() == CharLit2->getValue();
  }
  case Stmt::DeclRefExprClass: {
    const DeclRefExpr *DeclRef1 = cast<DeclRefExpr>(Expr1);
    const DeclRefExpr *DeclRef2 = cast<DeclRefExpr>(Expr2);
    return DeclRef1->getDecl() == DeclRef2->getDecl();
  }
  case Stmt::IntegerLiteralClass: {
    const IntegerLiteral *IntLit1 = cast<IntegerLiteral>(Expr1);
    const IntegerLiteral *IntLit2 = cast<IntegerLiteral>(Expr2);
    return IntLit1->getValue() == IntLit2->getValue();
  }
  case Stmt::FloatingLiteralClass: {
    const FloatingLiteral *FloatLit1 = cast<FloatingLiteral>(Expr1);
    const FloatingLiteral *FloatLit2 = cast<FloatingLiteral>(Expr2);
    return FloatLit1->getValue().bitwiseIsEqual(FloatLit2->getValue());
  }
  case Stmt::MemberExprClass: {
    const MemberExpr *MemberExpr1 = cast<MemberExpr>(Expr1);
    const MemberExpr *MemberExpr2 = cast<MemberExpr>(Expr2);
    return MemberExpr1->getMemberDecl() == MemberExpr2->getMemberDecl();
  }
  case Stmt::UnaryOperatorClass: {
    const UnaryOperator *UnaryOp1 = cast<UnaryOperator>(Expr1);
    const UnaryOperator *UnaryOp2 = cast<UnaryOperator>(Expr2);
    return UnaryOp1->getOpcode() == UnaryOp2->getOpcode();
  }
  }
}

//===----------------------------------------------------------------------===//
// FindIdenticalExprChecker
//===----------------------------------------------------------------------===//

namespace {
class FindIdenticalExprChecker : public Checker<check::ASTCodeBody> {
public:
  void checkASTCodeBody(const Decl *D, AnalysisManager &Mgr,
                        BugReporter &BR) const {
    FindIdenticalExprVisitor Visitor(BR, this, Mgr.getAnalysisDeclContext(D));
    Visitor.TraverseDecl(const_cast<Decl *>(D));
  }
};
} // end anonymous namespace

void ento::registerIdenticalExprChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<FindIdenticalExprChecker>();
}
