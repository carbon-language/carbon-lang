// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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
#include "../common/fortran.h"
#include "../common/indirection.h"
#include "../evaluate/expression.h"
#include "../evaluate/tools.h"
#include "../evaluate/type.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <optional>
#include <variant>

using namespace Fortran::parser::literals;

namespace Fortran::parser {
struct SourceLocationFindingVisitor {
  template<typename A> bool Pre(const A &) { return true; }
  template<typename A> void Post(const A &) {}
  bool Pre(const Expr &);
  template<typename A> bool Pre(const Statement<A> &stmt) {
    source = stmt.source;
    return false;
  }
  void Post(const CharBlock &);

  CharBlock source;
};

template<typename A> CharBlock FindSourceLocation(const A &x) {
  SourceLocationFindingVisitor visitor;
  Walk(x, visitor);
  return visitor.source;
}
}

using namespace Fortran::parser::literals;

// The expression semantic analysis code has its implementation in
// namespace Fortran::evaluate, but the exposed API to it is in the
// namespace Fortran::semantics (below).
//
// The template function AnalyzeExpr() is an internal interface
// between the implementation and the API used by semantic analysis.
// This template function has a few specializations here in the header
// file to handle what semantics might want to pass in as a top-level
// expression; other specializations appear in the implementation.
//
// The ExpressionAnalysisContext wraps a SemanticsContext reference
// and implements constraint checking on expressions using the
// parse tree node wrappers that mirror the grammar annotations used
// in the Fortran standard (i.e., scalar-, constant-, &c.).

namespace Fortran::evaluate {
class ExpressionAnalysisContext {
public:
  explicit ExpressionAnalysisContext(semantics::SemanticsContext &sc)
    : context_{sc} {}

  semantics::SemanticsContext &context() const { return context_; }

  FoldingContext &GetFoldingContext() const {
    return context_.foldingContext();
  }

  parser::ContextualMessages &GetContextualMessages() {
    return GetFoldingContext().messages;
  }

  template<typename... A> void Say(A... args) {
    GetContextualMessages().Say(std::forward<A>(args)...);
  }

  template<typename T, typename... A> void SayAt(const T &parsed, A... args) {
    Say(parser::FindSourceLocation(parsed), std::forward<A>(args)...);
  }

  std::optional<Expr<SomeType>> Analyze(const parser::Expr &);
  std::optional<Expr<SomeType>> Analyze(const parser::Variable &);
  Expr<SubscriptInteger> Analyze(common::TypeCategory category,
      const std::optional<parser::KindSelector> &);

  int GetDefaultKind(common::TypeCategory);
  DynamicType GetDefaultKindOfType(common::TypeCategory);

private:
  semantics::SemanticsContext &context_;
};

template<typename PARSED>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &, const PARSED &);

inline std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr &expr) {
  return context.Analyze(expr);
}
inline std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Variable &variable) {
  return context.Analyze(variable);
}

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

// These specializations implement constraint checking.

template<typename A>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Scalar<A> &x) {
  auto result{AnalyzeExpr(context, x.thing)};
  if (result.has_value()) {
    if (int rank{result->Rank()}; rank != 0) {
      context.SayAt(
          x, "Must be a scalar value, but is a rank-%d array"_err_en_US, rank);
    }
  }
  return result;
}

template<typename A>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Constant<A> &x) {
  auto result{AnalyzeExpr(context, x.thing)};
  if (result.has_value()) {
    *result = Fold(context.GetFoldingContext(), std::move(*result));
    if (!IsConstantExpr(*result)) {
      context.SayAt(x, "Must be a constant value"_err_en_US);
    }
  }
  return result;
}

template<typename A>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Integer<A> &x) {
  auto result{AnalyzeExpr(context, x.thing)};
  if (result.has_value()) {
    if (!std::holds_alternative<Expr<SomeInteger>>(result->u)) {
      context.SayAt(x, "Must have INTEGER type"_err_en_US);
    }
  }
  return result;
}

template<typename A>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Logical<A> &x) {
  auto result{AnalyzeExpr(context, x.thing)};
  if (result.has_value()) {
    if (!std::holds_alternative<Expr<SomeLogical>>(result->u)) {
      context.SayAt(x, "Must have LOGICAL type"_err_en_US);
    }
  }
  return result;
}
template<typename A>
std::optional<Expr<SomeType>> AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::DefaultChar<A> &x) {
  auto result{AnalyzeExpr(context, x.thing)};
  if (result.has_value()) {
    if (auto *charExpr{std::get_if<Expr<SomeCharacter>>(&result->u)}) {
      if (charExpr->GetKind() ==
          context.context().defaultKinds().GetDefaultKind(
              TypeCategory::Character)) {
        return result;
      }
    }
    context.SayAt(x, "Must have default CHARACTER type"_err_en_US);
  }
  return result;
}

template<typename L, typename R>
bool AreConformable(const L &left, const R &right) {
  int leftRank{left.Rank()};
  if (leftRank == 0) {
    return true;
  }
  int rightRank{right.Rank()};
  return rightRank == 0 || leftRank == rightRank;
}

template<typename L, typename R>
void ConformabilityCheck(
    parser::ContextualMessages &context, const L &left, const R &right) {
  if (!AreConformable(left, right)) {
    context.Say("left operand has rank %d, right operand has rank %d"_err_en_US,
        left.Rank(), right.Rank());
  }
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

// Semantic analysis of an intrinsic type's KIND parameter expression.
evaluate::Expr<evaluate::SubscriptInteger> AnalyzeKindSelector(
    SemanticsContext &, parser::CharBlock, common::TypeCategory,
    const std::optional<parser::KindSelector> &);
}
#endif  // FORTRAN_SEMANTICS_EXPRESSION_H_
