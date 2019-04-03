// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

// GetShape() analyzes an expression and determines its shape, if possible,
// representing the result as a vector of scalar integer expressions.

#ifndef FORTRAN_EVALUATE_SHAPE_H_
#define FORTRAN_EVALUATE_SHAPE_H_

#include "expression.h"
#include "type.h"
#include "../common/indirection.h"
#include <optional>
#include <variant>

namespace Fortran::evaluate {

using Extent = std::optional<Expr<SubscriptInteger>>;
using Shape = std::vector<Extent>;

template<typename A> std::optional<Shape> GetShape(const A &) {
  return std::nullopt;  // default case
}

// Forward declarations
template<typename... A>
std::optional<Shape> GetShape(const std::variant<A...> &);
template<typename A, bool COPY>
std::optional<Shape> GetShape(const common::Indirection<A, COPY> &);
template<typename A> std::optional<Shape> GetShape(const std::optional<A> &);

template<typename T> std::optional<Shape> GetShape(const Expr<T> &expr) {
  return GetShape(expr.u);
}

std::optional<Shape> GetShape(
    const semantics::Symbol &, const Component * = nullptr);
std::optional<Shape> GetShape(const BaseObject &);
std::optional<Shape> GetShape(const Component &);
std::optional<Shape> GetShape(const ArrayRef &);
std::optional<Shape> GetShape(const CoarrayRef &);
std::optional<Shape> GetShape(const DataRef &);
std::optional<Shape> GetShape(const Substring &);
std::optional<Shape> GetShape(const ComplexPart &);
std::optional<Shape> GetShape(const ActualArgument &);
std::optional<Shape> GetShape(const ProcedureRef &);
std::optional<Shape> GetShape(const StructureConstructor &);
std::optional<Shape> GetShape(const BOZLiteralConstant &);
std::optional<Shape> GetShape(const NullPointer &);

template<typename T>
std::optional<Shape> GetShape(const Designator<T> &designator) {
  return GetShape(designator.u);
}

template<typename T>
std::optional<Shape> GetShape(const Variable<T> &variable) {
  return GetShape(variable.u);
}

template<typename D, typename R, typename... O>
std::optional<Shape> GetShape(const Operation<D, R, O...> &operation) {
  if constexpr (operation.operands > 1) {
    if (operation.right().Rank() > 0) {
      return GetShape(operation.right());
    }
  }
  return GetShape(operation.left());
}

template<int KIND>
std::optional<Shape> GetShape(const TypeParamInquiry<KIND> &) {
  return Shape{};  // always scalar
}

template<typename T>
std::optional<Shape> GetShape(const ArrayConstructorValues<T> &aconst) {
  return std::nullopt;  // TODO pmk much more here!!
}

template<typename... A>
std::optional<Shape> GetShape(const std::variant<A...> &u) {
  return std::visit([](const auto &x) { return GetShape(x); }, u);
}

template<typename A, bool COPY>
std::optional<Shape> GetShape(const common::Indirection<A, COPY> &p) {
  return GetShape(p.value());
}

template<typename A> std::optional<Shape> GetShape(const std::optional<A> &x) {
  if (x.has_value()) {
    return GetShape(*x);
  } else {
    return std::nullopt;
  }
}
}
#endif  // FORTRAN_EVALUATE_SHAPE_H_
