//=-- CoverageMapping.cpp - Code coverage mapping support ---------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for clang's and llvm's instrumentation based
// code coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/CoverageMapping.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace coverage;

CounterExpressionBuilder::CounterExpressionBuilder(unsigned NumCounterValues) {
  Terms.resize(NumCounterValues);
}

Counter CounterExpressionBuilder::get(const CounterExpression &E) {
  for (unsigned I = 0, S = Expressions.size(); I < S; ++I) {
    if (Expressions[I] == E)
      return Counter::getExpression(I);
  }
  Expressions.push_back(E);
  return Counter::getExpression(Expressions.size() - 1);
}

void CounterExpressionBuilder::extractTerms(Counter C, int Sign) {
  switch (C.getKind()) {
  case Counter::Zero:
    break;
  case Counter::CounterValueReference:
    Terms[C.getCounterID()] += Sign;
    break;
  case Counter::Expression:
    const auto &E = Expressions[C.getExpressionID()];
    extractTerms(E.LHS, Sign);
    extractTerms(E.RHS, E.Kind == CounterExpression::Subtract ? -Sign : Sign);
    break;
  }
}

Counter CounterExpressionBuilder::simplify(Counter ExpressionTree) {
  // Gather constant terms.
  for (auto &I : Terms)
    I = 0;
  extractTerms(ExpressionTree);

  Counter C;
  // Create additions.
  // Note: the additions are created first
  // to avoid creation of a tree like ((0 - X) + Y) instead of (Y - X).
  for (unsigned I = 0, S = Terms.size(); I < S; ++I) {
    if (Terms[I] <= 0)
      continue;
    for (int J = 0; J < Terms[I]; ++J) {
      if (C.isZero())
        C = Counter::getCounter(I);
      else
        C = get(CounterExpression(CounterExpression::Add, C,
                                  Counter::getCounter(I)));
    }
  }

  // Create subtractions.
  for (unsigned I = 0, S = Terms.size(); I < S; ++I) {
    if (Terms[I] >= 0)
      continue;
    for (int J = 0; J < (-Terms[I]); ++J)
      C = get(CounterExpression(CounterExpression::Subtract, C,
                                Counter::getCounter(I)));
  }
  return C;
}

Counter CounterExpressionBuilder::add(Counter LHS, Counter RHS) {
  return simplify(get(CounterExpression(CounterExpression::Add, LHS, RHS)));
}

Counter CounterExpressionBuilder::subtract(Counter LHS, Counter RHS) {
  return simplify(
      get(CounterExpression(CounterExpression::Subtract, LHS, RHS)));
}

void CounterMappingContext::dump(const Counter &C,
                                 llvm::raw_ostream &OS) const {
  switch (C.getKind()) {
  case Counter::Zero:
    OS << '0';
    return;
  case Counter::CounterValueReference:
    OS << '#' << C.getCounterID();
    break;
  case Counter::Expression: {
    if (C.getExpressionID() >= Expressions.size())
      return;
    const auto &E = Expressions[C.getExpressionID()];
    OS << '(';
    dump(E.LHS, OS);
    OS << (E.Kind == CounterExpression::Subtract ? " - " : " + ");
    dump(E.RHS, OS);
    OS << ')';
    break;
  }
  }
  if (CounterValues.empty())
    return;
  ErrorOr<int64_t> Value = evaluate(C);
  if (!Value)
    return;
  OS << '[' << *Value << ']';
}

ErrorOr<int64_t> CounterMappingContext::evaluate(const Counter &C) const {
  switch (C.getKind()) {
  case Counter::Zero:
    return 0;
  case Counter::CounterValueReference:
    if (C.getCounterID() >= CounterValues.size())
      return std::errc::argument_out_of_domain;
    return CounterValues[C.getCounterID()];
  case Counter::Expression: {
    if (C.getExpressionID() >= Expressions.size())
      return std::errc::argument_out_of_domain;
    const auto &E = Expressions[C.getExpressionID()];
    ErrorOr<int64_t> LHS = evaluate(E.LHS);
    if (!LHS)
      return LHS;
    ErrorOr<int64_t> RHS = evaluate(E.RHS);
    if (!RHS)
      return RHS;
    return E.Kind == CounterExpression::Subtract ? *LHS - *RHS : *LHS + *RHS;
  }
  }
  llvm_unreachable("Unhandled CounterKind");
}
