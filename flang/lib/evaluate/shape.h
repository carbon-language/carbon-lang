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
#include "tools.h"
#include "type.h"
#include "../common/indirection.h"
#include <optional>
#include <variant>

namespace Fortran::evaluate {

using ExtentType = SubscriptInteger;
using ExtentExpr = Expr<ExtentType>;
using MaybeExtent = std::optional<ExtentExpr>;
using Shape = std::vector<MaybeExtent>;

// Convert a constant shape to the expression form, and vice versa.
Shape AsGeneralShape(const Constant<ExtentType> &);
std::optional<ExtentExpr> AsShapeArrayExpr(const Shape &);  // array constructor
std::optional<Constant<ExtentType>> AsConstantShape(const Shape &);

// Compute an element count for a triplet or trip count for a DO.
ExtentExpr CountTrips(
    ExtentExpr &&lower, ExtentExpr &&upper, ExtentExpr &&stride);
ExtentExpr CountTrips(
    const ExtentExpr &lower, const ExtentExpr &upper, const ExtentExpr &stride);
MaybeExtent CountTrips(
    MaybeExtent &&lower, MaybeExtent &&upper, MaybeExtent &&stride);

// Computes SIZE() == PRODUCT(shape)
MaybeExtent GetSize(Shape &&);

// Forward declarations
template<typename... A>
std::optional<Shape> GetShape(const std::variant<A...> &);
template<typename A, bool COPY>
std::optional<Shape> GetShape(const common::Indirection<A, COPY> &);
template<typename A> std::optional<Shape> GetShape(const std::optional<A> &);

template<typename T> std::optional<Shape> GetShape(const Expr<T> &expr) {
  return GetShape(expr.u);
}

std::optional<Shape> GetShape(const Symbol &, const Component * = nullptr);
std::optional<Shape> GetShape(const Symbol *);
std::optional<Shape> GetShape(const BaseObject &);
std::optional<Shape> GetShape(const Component &);
std::optional<Shape> GetShape(const ArrayRef &);
std::optional<Shape> GetShape(const CoarrayRef &);
std::optional<Shape> GetShape(const DataRef &);
std::optional<Shape> GetShape(const Substring &);
std::optional<Shape> GetShape(const ComplexPart &);
std::optional<Shape> GetShape(const ActualArgument &);
std::optional<Shape> GetShape(const ProcedureRef &);
std::optional<Shape> GetShape(const ImpliedDoIndex &);
std::optional<Shape> GetShape(const Relational<SomeType> &);
std::optional<Shape> GetShape(const StructureConstructor &);
std::optional<Shape> GetShape(const DescriptorInquiry &);
std::optional<Shape> GetShape(const BOZLiteralConstant &);
std::optional<Shape> GetShape(const NullPointer &);

template<typename T> std::optional<Shape> GetShape(const Constant<T> &c) {
  Constant<ExtentType> shape{c.SHAPE()};
  return AsGeneralShape(shape);
}

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
  if constexpr (sizeof...(O) > 1) {
    if (operation.right().Rank() > 0) {
      return GetShape(operation.right());
    }
  }
  return GetShape(operation.left());
}

template<int KIND>
std::optional<Shape> GetShape(const TypeParamInquiry<KIND> &) {
  return Shape{};  // always scalar, even when applied to an array
}

// Utility predicate: does an expression reference any implied DO index?
bool ContainsAnyImpliedDoIndex(const ExtentExpr &);

template<typename T> MaybeExtent GetExtent(const ArrayConstructorValues<T> &);

template<typename T>
MaybeExtent GetExtent(const ArrayConstructorValue<T> &value) {
  return std::visit(
      common::visitors{
          [](const common::CopyableIndirection<Expr<T>> &x) -> MaybeExtent {
            if (std::optional<Shape> xShape{GetShape(x)}) {
              // Array values in array constructors get linearized.
              return GetSize(std::move(*xShape));
            }
            return std::nullopt;
          },
          [](const ImpliedDo<T> &ido) -> MaybeExtent {
            // Don't be heroic and try to figure out triangular implied DO
            // nests.
            if (!ContainsAnyImpliedDoIndex(ido.lower()) &&
                !ContainsAnyImpliedDoIndex(ido.upper()) &&
                !ContainsAnyImpliedDoIndex(ido.stride())) {
              if (auto nValues{GetExtent(ido.values())}) {
                return std::move(*nValues) *
                    CountTrips(ido.lower(), ido.upper(), ido.stride());
              }
            }
            return std::nullopt;
          },
      },
      value.u);
}

template<typename T>
MaybeExtent GetExtent(const ArrayConstructorValues<T> &values) {
  ExtentExpr result{0};
  for (const auto &value : values.values()) {
    if (MaybeExtent n{GetExtent(value)}) {
      result = std::move(result) + std::move(*n);
    } else {
      return std::nullopt;
    }
  }
  return result;
}

template<typename T>
std::optional<Shape> GetShape(const ArrayConstructor<T> &aconst) {
  return Shape{GetExtent(aconst)};
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
