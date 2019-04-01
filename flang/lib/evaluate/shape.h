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
  return std::nullopt;
}

template<typename T> std::optional<Shape> GetShape(const Expr<T> &);

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

template<typename... A>
std::optional<Shape> GetShape(const std::variant<A...> &u) {
  return std::visit([](const auto &x) { return GetShape(x); }, u);
}

std::optional<Shape> GetShape(
    const semantics::Symbol &, const Component * = nullptr);
std::optional<Shape> GetShape(const DataRef &);
std::optional<Shape> GetShape(const ComplexPart &);
std::optional<Shape> GetShape(const Substring &);
std::optional<Shape> GetShape(const Component &);
std::optional<Shape> GetShape(const ArrayRef &);
std::optional<Shape> GetShape(const CoarrayRef &);

template<typename T>
std::optional<Shape> GetShape(const Designator<T> &designator) {
  return std::visit([](const auto &x) { return GetShape(x); }, designator.u);
}

template<typename T> std::optional<Shape> GetShape(const Expr<T> &expr) {
  return std::visit(
      common::visitors{
          [](const BOZLiteralConstant &) { return Shape{}; },
          [](const NullPointer &) { return std::nullopt; },
          [](const auto &x) { return GetShape(x); },
      },
      expr.u);
}
}
#endif  // FORTRAN_EVALUATE_SHAPE_H_
