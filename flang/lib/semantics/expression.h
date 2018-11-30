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
class ExpressionAnalysisContext {
public:
  using ConstraintChecker = bool (ExpressionAnalysisContext::*)(
      Expr<SomeType> &);

  ExpressionAnalysisContext(semantics::SemanticsContext &sc) : context_{sc} {}
  ExpressionAnalysisContext(ExpressionAnalysisContext &i)
    : context_{i.context_}, inner_{&i} {}
  ExpressionAnalysisContext(ExpressionAnalysisContext &i, ConstraintChecker cc)
    : context_{i.context_}, inner_{&i}, constraint_{cc} {}

  semantics::SemanticsContext &context() const { return context_; }

  template<typename... A> void Say(A... args) {
    context_.foldingContext().messages.Say(std::forward<A>(args)...);
  }

  void CheckConstraints(std::optional<Expr<SomeType>> &);
  bool ScalarConstraint(Expr<SomeType> &);
  bool ConstantConstraint(Expr<SomeType> &);
  bool IntegerConstraint(Expr<SomeType> &);

protected:
  semantics::SemanticsContext &context_;

private:
  ExpressionAnalysisContext *inner_{nullptr};
  ConstraintChecker constraint_{nullptr};
};

template<typename PARSED>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &, const PARSED &);

extern template std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr &);

template<typename A>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Scalar<A> &expr) {
  ExpressionAnalysisContext withCheck{
      context, &ExpressionAnalysisContext::ScalarConstraint};
  return AnalyzeExpr(withCheck, expr.thing);
}

template<typename A>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Constant<A> &expr) {
  ExpressionAnalysisContext withCheck{
      context, &ExpressionAnalysisContext::ConstantConstraint};
  return AnalyzeExpr(withCheck, expr.thing);
}

template<typename A>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Integer<A> &expr) {
  ExpressionAnalysisContext withCheck{
      context, &ExpressionAnalysisContext::ConstantConstraint};
  return AnalyzeExpr(withCheck, expr.thing);
}
}

namespace Fortran::semantics {

// Semantic analysis of one expression.

template<typename A>
std::optional<evaluate::Expr<evaluate::SomeType>> AnalyzeExpr(
    SemanticsContext &context, const A &expr) {
  evaluate::ExpressionAnalysisContext exprContext{context};
  return AnalyzeExpr(exprContext, expr);
}

// Semantic analysis of all expressions in a parse tree, which is
// decorated with typed representations for top-level expressions.
void AnalyzeExpressions(parser::Program &, SemanticsContext &);
}
#endif  // FORTRAN_SEMANTICS_EXPRESSION_H_
