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

#ifndef FORTRAN_EVALUATE_FOLD_H_
#define FORTRAN_EVALUATE_FOLD_H_

// Implements expression tree rewriting, particularly constant expression
// evaluation.

#include "common.h"
#include "constant.h"
#include "expression.h"
#include "tools.h"
#include "type.h"
#include <variant>

namespace Fortran::evaluate {

using namespace Fortran::parser::literals;

// Fold() rewrites an expression and returns it.  When the rewritten expression
// is a constant, GetScalarConstantValue() below will be able to extract it.
// Note the rvalue reference argument: the rewrites are performed in place
// for efficiency.
template<typename T> Expr<T> Fold(FoldingContext &context, Expr<T> &&expr) {
  return Expr<T>::Rewrite(context, std::move(expr));
}

template<typename T>
std::optional<Expr<T>> Fold(
    FoldingContext &context, std::optional<Expr<T>> &&expr) {
  if (expr.has_value()) {
    return {Fold(context, std::move(*expr))};
  } else {
    return std::nullopt;
  }
}

// GetScalarConstantValue() isolates the known constant value of
// an expression, if it has one.  The value can be parenthesized.
template<typename T>
std::optional<Scalar<T>> GetScalarConstantValue(const Expr<T> &expr) {
  if (const auto *c{UnwrapExpr<Constant<T>>(expr)}) {
    if (c->size() == 1) {
      return **c;
    } else {
      return std::nullopt;
    }
  } else if (const auto *parens{UnwrapExpr<Parentheses<T>>(expr)}) {
    return GetScalarConstantValue<T>(parens->left());
  } else {
    return std::nullopt;
  }
}

template<typename T>
std::optional<Scalar<T>> GetScalarConstantValue(
    const Expr<SomeKind<T::category>> &expr) {
  if (const auto *kindExpr{UnwrapExpr<Expr<T>>(expr)}) {
    return GetScalarConstantValue<T>(*kindExpr);
  }
  return std::nullopt;
}

template<typename T>
std::optional<Scalar<T>> GetScalarConstantValue(const Expr<SomeType> &expr) {
  if (const auto *kindExpr{UnwrapExpr<Expr<T>>(expr)}) {
    return GetScalarConstantValue<T>(*kindExpr);
  }
  return std::nullopt;
}

template<typename T, typename A>
std::optional<Scalar<T>> GetScalarConstantValue(const std::optional<A> &expr) {
  if (expr.has_value()) {
    return GetScalarConstantValue(*expr);
  } else {
    return std::nullopt;
  }
}

// Predicate: true when an expression is a constant expression (in the
// strict sense of the Fortran standard); it may not (yet) be a hard
// constant value.
bool IsConstantExpr(const Expr<SomeType> &);

// When an expression is a constant integer, ToInt64() extracts its value.
// Ensure that the expression has been folded beforehand when folding might
// be required.
template<int KIND>
std::optional<std::int64_t> ToInt64(
    const Expr<Type<TypeCategory::Integer, KIND>> &expr) {
  if (auto scalar{GetScalarConstantValue(expr)}) {
    return {scalar->ToInt64()};
  } else {
    return std::nullopt;
  }
}

std::optional<std::int64_t> ToInt64(const Expr<SomeInteger> &);
std::optional<std::int64_t> ToInt64(const Expr<SomeType> &);

template<typename A>
std::optional<std::int64_t> ToInt64(const std::optional<A> &x) {
  if (x.has_value()) {
    return ToInt64(*x);
  } else {
    return std::nullopt;
  }
}
}
#endif  // FORTRAN_EVALUATE_FOLD_H_
