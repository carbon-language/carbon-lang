// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FORTRAN_SEMANTICS_EXPRESSION_H_
#define FORTRAN_SEMANTICS_EXPRESSION_H_

#include "semantics.h"
#include "../evaluate/expression.h"
#include "../evaluate/type.h"
#include <optional>
#include <variant>

namespace Fortran::parser {
struct Expr;
struct Program;
template<typename> struct Scalar;
template<typename> struct Integer;
template<typename> struct Constant;
}

namespace Fortran::evaluate {
struct Constraints;
struct ExpressionAnalysisContext {
  using ConstraintChecker = bool (ExpressionAnalysisContext::*)(
      Expr<SomeType> &);

  ExpressionAnalysisContext(semantics::SemanticsContext &ctx) : context{ctx} {}

  template<typename... A> void Say(A... args) {
    context.foldingContext().messages.Say(std::forward<A>(args)...);
  }

  void CheckConstraints(std::optional<Expr<SomeType>> &, const Constraints *);
  bool ScalarConstraint(Expr<SomeType> &);
  bool ConstantConstraint(Expr<SomeType> &);
  bool IntegerConstraint(Expr<SomeType> &);

  semantics::SemanticsContext &context;
};

// Constraint checking (e.g., for Scalar<> expressions) is implemented by
// passing a pointer to one of these partial closures along to AnalyzeExpr.
// The constraint can then be checked and errors reported with precise
// source program location information.
struct Constraints {
  ExpressionAnalysisContext::ConstraintChecker checker;
  const Constraints *inner{nullptr};
};
}

namespace Fortran::semantics {

class SemanticsContext;

// Semantic analysis of one expression.

template<typename PARSED>
std::optional<evaluate::Expr<evaluate::SomeType>> AnalyzeExpr(
    SemanticsContext &, const PARSED &,
    const evaluate::Constraints * = nullptr);

extern template std::optional<evaluate::Expr<evaluate::SomeType>> AnalyzeExpr(
    SemanticsContext &, const parser::Expr &,
    const evaluate::Constraints *c = nullptr);

template<typename A>
std::optional<evaluate::Expr<evaluate::SomeType>> AnalyzeExpr(
    SemanticsContext &context, const parser::Scalar<A> &expr,
    const evaluate::Constraints *constraints = nullptr) {
  evaluate::Constraints newConstraints{
      &evaluate::ExpressionAnalysisContext::ScalarConstraint, constraints};
  return AnalyzeExpr(context, expr.thing, &newConstraints);
}

template<typename A>
std::optional<evaluate::Expr<evaluate::SomeType>> AnalyzeExpr(
    SemanticsContext &context, const parser::Constant<A> &expr,
    const evaluate::Constraints *constraints = nullptr) {
  evaluate::Constraints newConstraints{
      &evaluate::ExpressionAnalysisContext::ConstantConstraint, constraints};
  return AnalyzeExpr(context, expr.thing, &newConstraints);
}

template<typename A>
std::optional<evaluate::Expr<evaluate::SomeType>> AnalyzeExpr(
    SemanticsContext &context, const parser::Integer<A> &expr,
    const evaluate::Constraints *constraints = nullptr) {
  evaluate::Constraints newConstraints{
      &evaluate::ExpressionAnalysisContext::IntegerConstraint, constraints};
  return AnalyzeExpr(context, expr.thing, &newConstraints);
}

// Semantic analysis of all expressions in a parse tree, which is
// decorated with typed representations for top-level expressions.
void AnalyzeExpressions(parser::Program &, SemanticsContext &);
}
#endif  // FORTRAN_SEMANTICS_EXPRESSION_H_
