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

#ifndef FORTRAN_EVALUATE_FOLD_H_
#define FORTRAN_EVALUATE_FOLD_H_

// Implements expression tree rewriting, particularly constant expression
// evaluation.

#include "common.h"
#include "expression.h"
#include "type.h"

namespace Fortran::evaluate {

using namespace Fortran::parser::literals;

// Fold() rewrites an expression and returns it.  When the rewritten expression
// is a constant, GetScalarConstantValue() below will be able to extract it.
// Note the rvalue reference argument: the rewrites are performed in place
// for efficiency.  The implementation is wrapped in a helper template class so
// that all the per-type template instantiations can be made once in fold.cc.
template<typename T> struct FoldHelper {
  static Expr<T> FoldExpr(FoldingContext &, Expr<T> &&);
};

template<typename T> Expr<T> Fold(FoldingContext &context, Expr<T> &&expr) {
  return FoldHelper<T>::FoldExpr(context, std::move(expr));
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

FOR_EACH_TYPE_AND_KIND(extern template struct FoldHelper, ;)

// GetScalarConstantValue() extracts the constant value of an expression,
// when it has one, even if it is parenthesized or optional.
template<typename T> struct GetScalarConstantValueHelper {
  static std::optional<Constant<T>> GetScalarConstantValue(const Expr<T> &);
};

template<typename T>
std::optional<Constant<T>> GetScalarConstantValue(const Expr<T> &expr) {
  return GetScalarConstantValueHelper<T>::GetScalarConstantValue(expr);
}
template<typename T>
std::optional<Constant<T>> GetScalarConstantValue(
    const std::optional<Expr<T>> &expr) {
  if (expr.has_value()) {
    return GetScalarConstantValueHelper<T>::GetScalarConstantValue(*expr);
  } else {
    return std::nullopt;
  }
}

FOR_EACH_INTRINSIC_KIND(extern template struct GetScalarConstantValueHelper, ;)

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_FOLD_H_
