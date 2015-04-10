//===--- SimplifyBooleanExpr.cpp clang-tidy ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SimplifyBooleanExprCheck.h"
#include "clang/Lex/Lexer.h"

#include <cassert>
#include <string>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

namespace {

StringRef getText(const MatchFinder::MatchResult &Result, SourceRange Range) {
  return Lexer::getSourceText(CharSourceRange::getTokenRange(Range),
                              *Result.SourceManager,
                              Result.Context->getLangOpts());
}

template <typename T>
StringRef getText(const MatchFinder::MatchResult &Result, T &Node) {
  return getText(Result, Node.getSourceRange());
}

const char RightExpressionId[] = "bool-op-expr-yields-expr";
const char LeftExpressionId[] = "expr-op-bool-yields-expr";
const char NegatedRightExpressionId[] = "bool-op-expr-yields-not-expr";
const char NegatedLeftExpressionId[] = "expr-op-bool-yields-not-expr";
const char ConditionThenStmtId[] = "if-bool-yields-then";
const char ConditionElseStmtId[] = "if-bool-yields-else";
const char TernaryId[] = "ternary-bool-yields-condition";
const char TernaryNegatedId[] = "ternary-bool-yields-not-condition";
const char IfReturnsBoolId[] = "if-return";
const char IfReturnsNotBoolId[] = "if-not-return";
const char ThenLiteralId[] = "then-literal";
const char IfAssignVariableId[] = "if-assign-lvalue";
const char IfAssignLocId[] = "if-assign-loc";
const char IfAssignBoolId[] = "if-assign";
const char IfAssignNotBoolId[] = "if-assign-not";
const char IfAssignObjId[] = "if-assign-obj";

const char IfStmtId[] = "if";
const char LHSId[] = "lhs-expr";
const char RHSId[] = "rhs-expr";

const char SimplifyOperatorDiagnostic[] =
    "redundant boolean literal supplied to boolean operator";
const char SimplifyConditionDiagnostic[] =
    "redundant boolean literal in if statement condition";

const CXXBoolLiteralExpr *getBoolLiteral(const MatchFinder::MatchResult &Result,
                                         StringRef Id) {
  const auto *Literal = Result.Nodes.getNodeAs<CXXBoolLiteralExpr>(Id);
  return (Literal &&
          Result.SourceManager->isMacroBodyExpansion(Literal->getLocStart()))
             ? nullptr
             : Literal;
}

internal::Matcher<Stmt> ReturnsBool(bool Value, StringRef Id = "") {
  auto SimpleReturnsBool = returnStmt(
      has(boolLiteral(equals(Value)).bind(Id.empty() ? "ignored" : Id)));
  return anyOf(SimpleReturnsBool,
               compoundStmt(statementCountIs(1), has(SimpleReturnsBool)));
}

bool needsParensAfterUnaryNegation(const Expr *E) {
  if (isa<BinaryOperator>(E) || isa<ConditionalOperator>(E))
    return true;
  if (const auto *Op = dyn_cast<CXXOperatorCallExpr>(E))
    return Op->getNumArgs() == 2 && Op->getOperator() != OO_Call &&
           Op->getOperator() != OO_Subscript;
  return false;
}

std::pair<BinaryOperatorKind, BinaryOperatorKind> Opposites[] = {
    std::make_pair(BO_LT, BO_GE), std::make_pair(BO_GT, BO_LE),
    std::make_pair(BO_EQ, BO_NE)};

StringRef negatedOperator(const BinaryOperator *BinOp) {
  const BinaryOperatorKind Opcode = BinOp->getOpcode();
  for (auto NegatableOp : Opposites) {
    if (Opcode == NegatableOp.first)
      return BinOp->getOpcodeStr(NegatableOp.second);
    if (Opcode == NegatableOp.second)
      return BinOp->getOpcodeStr(NegatableOp.first);
  }
  return StringRef();
}

std::string replacementExpression(const MatchFinder::MatchResult &Result,
                                  bool Negated, const Expr *E) {
  while (const auto *Parenthesized = dyn_cast<ParenExpr>(E)) {
    E = Parenthesized->getSubExpr();
  }
  if (Negated) {
    if (const auto *BinOp = dyn_cast<BinaryOperator>(E)) {
      StringRef NegatedOperator = negatedOperator(BinOp);
      if (!NegatedOperator.empty()) {
        return (getText(Result, *BinOp->getLHS()) + " " + NegatedOperator +
                " " + getText(Result, *BinOp->getRHS()))
            .str();
      }
    }
  }
  StringRef Text = getText(Result, *E);
  return (Negated ? (needsParensAfterUnaryNegation(E) ? "!(" + Text + ")"
                                                      : "!" + Text)
                  : Text)
      .str();
}

} // namespace

void SimplifyBooleanExprCheck::matchBoolBinOpExpr(MatchFinder *Finder,
                                                  bool Value,
                                                  StringRef OperatorName,
                                                  StringRef BooleanId) {
  Finder->addMatcher(
      binaryOperator(isExpansionInMainFile(), hasOperatorName(OperatorName),
                     hasLHS(allOf(expr().bind(LHSId),
                                  boolLiteral(equals(Value)).bind(BooleanId))),
                     hasRHS(expr().bind(RHSId)),
                     unless(hasRHS(hasDescendant(boolLiteral())))),
      this);
}

void SimplifyBooleanExprCheck::matchExprBinOpBool(MatchFinder *Finder,
                                                  bool Value,
                                                  StringRef OperatorName,
                                                  StringRef BooleanId) {
  Finder->addMatcher(
      binaryOperator(
          isExpansionInMainFile(), hasOperatorName(OperatorName),
          hasLHS(expr().bind(LHSId)),
          unless(hasLHS(anyOf(boolLiteral(), hasDescendant(boolLiteral())))),
          hasRHS(allOf(expr().bind(RHSId),
                       boolLiteral(equals(Value)).bind(BooleanId)))),
      this);
}

void SimplifyBooleanExprCheck::matchBoolCompOpExpr(MatchFinder *Finder,
                                                   bool Value,
                                                   StringRef OperatorName,
                                                   StringRef BooleanId) {
  Finder->addMatcher(
      binaryOperator(isExpansionInMainFile(), hasOperatorName(OperatorName),
                     hasLHS(allOf(expr().bind(LHSId),
                                  ignoringImpCasts(boolLiteral(equals(Value))
                                                       .bind(BooleanId)))),
                     hasRHS(expr().bind(RHSId)),
                     unless(hasRHS(hasDescendant(boolLiteral())))),
      this);
}

void SimplifyBooleanExprCheck::matchExprCompOpBool(MatchFinder *Finder,
                                                   bool Value,
                                                   StringRef OperatorName,
                                                   StringRef BooleanId) {
  Finder->addMatcher(
      binaryOperator(isExpansionInMainFile(), hasOperatorName(OperatorName),
                     unless(hasLHS(hasDescendant(boolLiteral()))),
                     hasLHS(expr().bind(LHSId)),
                     hasRHS(allOf(expr().bind(RHSId),
                                  ignoringImpCasts(boolLiteral(equals(Value))
                                                       .bind(BooleanId))))),
      this);
}

void SimplifyBooleanExprCheck::matchBoolCondition(MatchFinder *Finder,
                                                  bool Value,
                                                  StringRef BooleanId) {
  Finder->addMatcher(ifStmt(isExpansionInMainFile(),
                            hasCondition(boolLiteral(equals(Value))
                                             .bind(BooleanId))).bind(IfStmtId),
                     this);
}

void SimplifyBooleanExprCheck::matchTernaryResult(MatchFinder *Finder,
                                                  bool Value,
                                                  StringRef TernaryId) {
  Finder->addMatcher(
      conditionalOperator(isExpansionInMainFile(),
                          hasTrueExpression(boolLiteral(equals(Value))),
                          hasFalseExpression(boolLiteral(equals(!Value))))
          .bind(TernaryId),
      this);
}

void SimplifyBooleanExprCheck::matchIfReturnsBool(MatchFinder *Finder,
                                                  bool Value, StringRef Id) {
  Finder->addMatcher(ifStmt(isExpansionInMainFile(),
                            hasThen(ReturnsBool(Value, ThenLiteralId)),
                            hasElse(ReturnsBool(!Value))).bind(Id),
                     this);
}

void SimplifyBooleanExprCheck::matchIfAssignsBool(MatchFinder *Finder,
                                                  bool Value, StringRef Id) {
  auto SimpleThen = binaryOperator(
      hasOperatorName("="),
      hasLHS(declRefExpr(hasDeclaration(decl().bind(IfAssignObjId)))),
      hasLHS(expr().bind(IfAssignVariableId)),
      hasRHS(boolLiteral(equals(Value)).bind(IfAssignLocId)));
  auto Then = anyOf(SimpleThen, compoundStmt(statementCountIs(1),
                                             hasAnySubstatement(SimpleThen)));
  auto SimpleElse = binaryOperator(
      hasOperatorName("="),
      hasLHS(declRefExpr(hasDeclaration(equalsBoundNode(IfAssignObjId)))),
      hasRHS(boolLiteral(equals(!Value))));
  auto Else = anyOf(SimpleElse, compoundStmt(statementCountIs(1),
                                             hasAnySubstatement(SimpleElse)));
  Finder->addMatcher(
      ifStmt(isExpansionInMainFile(), hasThen(Then), hasElse(Else)).bind(Id),
      this);
}

void SimplifyBooleanExprCheck::registerMatchers(MatchFinder *Finder) {
  matchBoolBinOpExpr(Finder, true, "&&", RightExpressionId);
  matchBoolBinOpExpr(Finder, false, "||", RightExpressionId);
  matchExprBinOpBool(Finder, false, "&&", RightExpressionId);
  matchExprBinOpBool(Finder, true, "||", RightExpressionId);
  matchBoolCompOpExpr(Finder, true, "==", RightExpressionId);
  matchBoolCompOpExpr(Finder, false, "!=", RightExpressionId);

  matchExprBinOpBool(Finder, true, "&&", LeftExpressionId);
  matchExprBinOpBool(Finder, false, "||", LeftExpressionId);
  matchBoolBinOpExpr(Finder, false, "&&", LeftExpressionId);
  matchBoolBinOpExpr(Finder, true, "||", LeftExpressionId);
  matchExprCompOpBool(Finder, true, "==", LeftExpressionId);
  matchExprCompOpBool(Finder, false, "!=", LeftExpressionId);

  matchBoolCompOpExpr(Finder, false, "==", NegatedRightExpressionId);
  matchBoolCompOpExpr(Finder, true, "!=", NegatedRightExpressionId);

  matchExprCompOpBool(Finder, false, "==", NegatedLeftExpressionId);
  matchExprCompOpBool(Finder, true, "!=", NegatedLeftExpressionId);

  matchBoolCondition(Finder, true, ConditionThenStmtId);
  matchBoolCondition(Finder, false, ConditionElseStmtId);

  matchTernaryResult(Finder, true, TernaryId);
  matchTernaryResult(Finder, false, TernaryNegatedId);

  matchIfReturnsBool(Finder, true, IfReturnsBoolId);
  matchIfReturnsBool(Finder, false, IfReturnsNotBoolId);

  matchIfAssignsBool(Finder, true, IfAssignBoolId);
  matchIfAssignsBool(Finder, false, IfAssignNotBoolId);
}

void SimplifyBooleanExprCheck::check(const MatchFinder::MatchResult &Result) {
  if (const CXXBoolLiteralExpr *LeftRemoved =
          getBoolLiteral(Result, RightExpressionId)) {
    replaceWithExpression(Result, LeftRemoved, false);
  } else if (const CXXBoolLiteralExpr *RightRemoved =
                 getBoolLiteral(Result, LeftExpressionId)) {
    replaceWithExpression(Result, RightRemoved, true);
  } else if (const CXXBoolLiteralExpr *NegatedLeftRemoved =
                 getBoolLiteral(Result, NegatedRightExpressionId)) {
    replaceWithExpression(Result, NegatedLeftRemoved, false, true);
  } else if (const CXXBoolLiteralExpr *NegatedRightRemoved =
                 getBoolLiteral(Result, NegatedLeftExpressionId)) {
    replaceWithExpression(Result, NegatedRightRemoved, true, true);
  } else if (const CXXBoolLiteralExpr *TrueConditionRemoved =
                 getBoolLiteral(Result, ConditionThenStmtId)) {
    replaceWithThenStatement(Result, TrueConditionRemoved);
  } else if (const CXXBoolLiteralExpr *FalseConditionRemoved =
                 getBoolLiteral(Result, ConditionElseStmtId)) {
    replaceWithElseStatement(Result, FalseConditionRemoved);
  } else if (const auto *Ternary =
                 Result.Nodes.getNodeAs<ConditionalOperator>(TernaryId)) {
    replaceWithCondition(Result, Ternary);
  } else if (const auto *TernaryNegated =
                 Result.Nodes.getNodeAs<ConditionalOperator>(
                     TernaryNegatedId)) {
    replaceWithCondition(Result, TernaryNegated, true);
  } else if (const auto *If = Result.Nodes.getNodeAs<IfStmt>(IfReturnsBoolId)) {
    replaceWithReturnCondition(Result, If);
  } else if (const auto *IfNot =
                 Result.Nodes.getNodeAs<IfStmt>(IfReturnsNotBoolId)) {
    replaceWithReturnCondition(Result, IfNot, true);
  } else if (const auto *IfAssign =
                 Result.Nodes.getNodeAs<IfStmt>(IfAssignBoolId)) {
    replaceWithAssignment(Result, IfAssign);
  } else if (const auto *IfAssignNot =
                 Result.Nodes.getNodeAs<IfStmt>(IfAssignNotBoolId)) {
    replaceWithAssignment(Result, IfAssignNot, true);
  }
}

void SimplifyBooleanExprCheck::replaceWithExpression(
    const ast_matchers::MatchFinder::MatchResult &Result,
    const CXXBoolLiteralExpr *BoolLiteral, bool UseLHS, bool Negated) {
  const auto *LHS = Result.Nodes.getNodeAs<Expr>(LHSId);
  const auto *RHS = Result.Nodes.getNodeAs<Expr>(RHSId);
  std::string Replacement =
      replacementExpression(Result, Negated, UseLHS ? LHS : RHS);
  SourceLocation Start = LHS->getLocStart();
  SourceLocation End = RHS->getLocEnd();
  diag(BoolLiteral->getLocStart(), SimplifyOperatorDiagnostic)
      << FixItHint::CreateReplacement(SourceRange(Start, End), Replacement);
}

void SimplifyBooleanExprCheck::replaceWithThenStatement(
    const MatchFinder::MatchResult &Result,
    const CXXBoolLiteralExpr *TrueConditionRemoved) {
  const auto *IfStatement = Result.Nodes.getNodeAs<IfStmt>(IfStmtId);
  diag(TrueConditionRemoved->getLocStart(), SimplifyConditionDiagnostic)
      << FixItHint::CreateReplacement(IfStatement->getSourceRange(),
                                      getText(Result, *IfStatement->getThen()));
}

void SimplifyBooleanExprCheck::replaceWithElseStatement(
    const MatchFinder::MatchResult &Result,
    const CXXBoolLiteralExpr *FalseConditionRemoved) {
  const auto *IfStatement = Result.Nodes.getNodeAs<IfStmt>(IfStmtId);
  const Stmt *ElseStatement = IfStatement->getElse();
  diag(FalseConditionRemoved->getLocStart(), SimplifyConditionDiagnostic)
      << FixItHint::CreateReplacement(
          IfStatement->getSourceRange(),
          ElseStatement ? getText(Result, *ElseStatement) : "");
}

void SimplifyBooleanExprCheck::replaceWithCondition(
    const MatchFinder::MatchResult &Result, const ConditionalOperator *Ternary,
    bool Negated) {
  std::string Replacement =
      replacementExpression(Result, Negated, Ternary->getCond());
  diag(Ternary->getTrueExpr()->getLocStart(),
       "redundant boolean literal in ternary expression result")
      << FixItHint::CreateReplacement(Ternary->getSourceRange(), Replacement);
}

void SimplifyBooleanExprCheck::replaceWithReturnCondition(
    const MatchFinder::MatchResult &Result, const IfStmt *If, bool Negated) {
  StringRef Terminator = isa<CompoundStmt>(If->getElse()) ? ";" : "";
  std::string Condition = replacementExpression(Result, Negated, If->getCond());
  std::string Replacement = ("return " + Condition + Terminator).str();
  SourceLocation Start =
      Result.Nodes.getNodeAs<CXXBoolLiteralExpr>(ThenLiteralId)->getLocStart();
  diag(Start, "redundant boolean literal in conditional return statement")
      << FixItHint::CreateReplacement(If->getSourceRange(), Replacement);
}

void SimplifyBooleanExprCheck::replaceWithAssignment(
    const MatchFinder::MatchResult &Result, const IfStmt *IfAssign,
    bool Negated) {
  SourceRange Range = IfAssign->getSourceRange();
  StringRef VariableName =
      getText(Result, *Result.Nodes.getNodeAs<Expr>(IfAssignVariableId));
  StringRef Terminator = isa<CompoundStmt>(IfAssign->getElse()) ? ";" : "";
  std::string Condition =
      replacementExpression(Result, Negated, IfAssign->getCond());
  std::string Replacement =
      (VariableName + " = " + Condition + Terminator).str();
  SourceLocation Location =
      Result.Nodes.getNodeAs<CXXBoolLiteralExpr>(IfAssignLocId)->getLocStart();
  this->diag(Location, "redundant boolean literal in conditional assignment")
      << FixItHint::CreateReplacement(Range, Replacement);
}

} // namespace readability
} // namespace tidy
} // namespace clang
