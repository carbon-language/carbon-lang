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
template<typename> struct Logical;
template<typename> struct DefaultChar;
}

// The expression semantic analysis code has its implementation in
// namespace Fortran::evaluate, but the exposed API to it is in the
// namespace Fortran::semantics (below).
//
// The template function AnalyzeExpr is an internal interface
// between the implementation and the API used by semantic analysis.
// This template function has a few specializations here in the header
// file to handle what semantics might want to pass in as a top-level
// expression; other specializations appear in the implementation.
//
// The ExpressionAnalysisContext wraps a SemanticsContext reference
// and implements constraint checking on expressions using the
// parse tree node wrappers that mirror the grammar annotations used
// in the Fortran standard (i.e., scalar-, constant-, &c.).  These
// constraint checks are performed in a deferred manner so that any
// errors are reported on the most accurate source location available.

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
  bool LogicalConstraint(Expr<SomeType> &);
  bool DefaultCharConstraint(Expr<SomeType> &);

protected:
  semantics::SemanticsContext &context_;

private:
  ExpressionAnalysisContext *inner_{nullptr};
  ConstraintChecker constraint_{nullptr};
};

template<typename PARSED>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &, const PARSED &);

// This extern template is the gateway into the rest of the expression
// analysis implementation in expression.cc.
extern template std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr &);

// Forward declarations of exposed specializations
template<typename A>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &, const common::Indirection<A> &);
template<typename A>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Scalar<A> &);
template<typename A>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Constant<A> &);
template<typename A>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Integer<A> &);
template<typename A>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Logical<A> &);
template<typename A>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::DefaultChar<A> &);

// Indirections are silently traversed by AnalyzeExpr().
template<typename A>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &context, const common::Indirection<A> &x) {
  return AnalyzeExpr(context, *x);
}

// These specializations create nested expression analysis contexts
// to implement constraint checking.

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
      context, &ExpressionAnalysisContext::IntegerConstraint};
  return AnalyzeExpr(withCheck, expr.thing);
}

template<typename A>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Logical<A> &expr) {
  ExpressionAnalysisContext withCheck{
      context, &ExpressionAnalysisContext::LogicalConstraint};
  return AnalyzeExpr(withCheck, expr.thing);
}
template<typename A>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::DefaultChar<A> &expr) {
  ExpressionAnalysisContext withCheck{
      context, &ExpressionAnalysisContext::DefaultCharConstraint};
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
