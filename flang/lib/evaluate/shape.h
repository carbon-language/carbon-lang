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
#include "variable.h"
#include "../common/indirection.h"
#include <optional>
#include <variant>

namespace Fortran::parser {
class ContextualMessages;
}

namespace Fortran::evaluate {

class FoldingContext;

using ExtentType = SubscriptInteger;
using ExtentExpr = Expr<ExtentType>;
using MaybeExtentExpr = std::optional<ExtentExpr>;
using Shape = std::vector<MaybeExtentExpr>;

bool IsImpliedShape(const Symbol &);
bool IsExplicitShape(const Symbol &);

// Conversions between various representations of shapes.
Shape AsShape(const Constant<ExtentType> &);
std::optional<Shape> AsShape(FoldingContext &, ExtentExpr &&);

std::optional<ExtentExpr> AsExtentArrayExpr(const Shape &);

std::optional<Constant<ExtentType>> AsConstantShape(const Shape &);
Constant<ExtentType> AsConstantShape(const ConstantSubscripts &);

ConstantSubscripts AsConstantExtents(const Constant<ExtentType> &);
std::optional<ConstantSubscripts> AsConstantExtents(const Shape &);

inline int GetRank(const Shape &s) { return static_cast<int>(s.size()); }

// The dimension argument to these inquiries is zero-based,
// unlike the DIM= arguments to many intrinsics.
MaybeExtentExpr GetLowerBound(
    FoldingContext &, const NamedEntity &, int dimension);
Shape GetLowerBounds(FoldingContext &, const NamedEntity &);
MaybeExtentExpr GetExtent(FoldingContext &, const NamedEntity &, int dimension);
MaybeExtentExpr GetExtent(
    FoldingContext &, const Subscript &, const NamedEntity &, int dimension);
MaybeExtentExpr GetUpperBound(
    FoldingContext &, MaybeExtentExpr &&lower, MaybeExtentExpr &&extent);

// Compute an element count for a triplet or trip count for a DO.
ExtentExpr CountTrips(
    ExtentExpr &&lower, ExtentExpr &&upper, ExtentExpr &&stride);
ExtentExpr CountTrips(
    const ExtentExpr &lower, const ExtentExpr &upper, const ExtentExpr &stride);
MaybeExtentExpr CountTrips(
    MaybeExtentExpr &&lower, MaybeExtentExpr &&upper, MaybeExtentExpr &&stride);

// Computes SIZE() == PRODUCT(shape)
MaybeExtentExpr GetSize(Shape &&);

// Utility predicate: does an expression reference any implied DO index?
bool ContainsAnyImpliedDoIndex(const ExtentExpr &);

// Compilation-time shape conformance checking, when corresponding extents
// are known.
bool CheckConformance(parser::ContextualMessages &, const Shape &,
    const Shape &, const char * = "left operand",
    const char * = "right operand");

// The implementation of GetShape() is wrapped in a helper class
// so that the member functions may mutually recurse without prototypes.
class GetShapeHelper {
public:
  explicit GetShapeHelper(FoldingContext &context) : context_{context} {}

  template<typename T> std::optional<Shape> GetShape(const Expr<T> &expr) {
    return GetShape(expr.u);
  }

  std::optional<Shape> GetShape(const Symbol &);
  std::optional<Shape> GetShape(const Symbol *);
  std::optional<Shape> GetShape(const Component &);
  std::optional<Shape> GetShape(const NamedEntity &);
  std::optional<Shape> GetShape(const BaseObject &);
  std::optional<Shape> GetShape(const ArrayRef &);
  std::optional<Shape> GetShape(const CoarrayRef &);
  std::optional<Shape> GetShape(const DataRef &);
  std::optional<Shape> GetShape(const Substring &);
  std::optional<Shape> GetShape(const ComplexPart &);
  std::optional<Shape> GetShape(const ActualArgument &);
  std::optional<Shape> GetShape(const ProcedureDesignator &);
  std::optional<Shape> GetShape(const ProcedureRef &);
  std::optional<Shape> GetShape(const ImpliedDoIndex &);
  std::optional<Shape> GetShape(const Relational<SomeType> &);
  std::optional<Shape> GetShape(const StructureConstructor &);
  std::optional<Shape> GetShape(const DescriptorInquiry &);
  std::optional<Shape> GetShape(const BOZLiteralConstant &);
  std::optional<Shape> GetShape(const NullPointer &);

  template<typename T> std::optional<Shape> GetShape(const Constant<T> &c) {
    Constant<ExtentType> shape{c.SHAPE()};
    return AsShape(shape);
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

  template<typename T>
  std::optional<Shape> GetShape(const ArrayConstructor<T> &aconst) {
    return Shape{GetArrayConstructorExtent(aconst)};
  }

  template<typename... A>
  std::optional<Shape> GetShape(const std::variant<A...> &u) {
    return std::visit([&](const auto &x) { return GetShape(x); }, u);
  }

  template<typename A, bool COPY>
  std::optional<Shape> GetShape(const common::Indirection<A, COPY> &p) {
    return GetShape(p.value());
  }

  template<typename A>
  std::optional<Shape> GetShape(const std::optional<A> &x) {
    if (x.has_value()) {
      return GetShape(*x);
    } else {
      return std::nullopt;
    }
  }

private:
  template<typename T>
  MaybeExtentExpr GetArrayConstructorValueExtent(
      const ArrayConstructorValue<T> &value) {
    return std::visit(
        common::visitors{
            [&](const Expr<T> &x) -> MaybeExtentExpr {
              if (std::optional<Shape> xShape{GetShape(x)}) {
                // Array values in array constructors get linearized.
                return GetSize(std::move(*xShape));
              } else {
                return std::nullopt;
              }
            },
            [&](const ImpliedDo<T> &ido) -> MaybeExtentExpr {
              // Don't be heroic and try to figure out triangular implied DO
              // nests.
              if (!ContainsAnyImpliedDoIndex(ido.lower()) &&
                  !ContainsAnyImpliedDoIndex(ido.upper()) &&
                  !ContainsAnyImpliedDoIndex(ido.stride())) {
                if (auto nValues{GetArrayConstructorExtent(ido.values())}) {
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
  MaybeExtentExpr GetArrayConstructorExtent(
      const ArrayConstructorValues<T> &values) {
    ExtentExpr result{0};
    for (const auto &value : values) {
      if (MaybeExtentExpr n{GetArrayConstructorValueExtent(value)}) {
        result = std::move(result) + std::move(*n);
      } else {
        return std::nullopt;
      }
    }
    return result;
  }

  FoldingContext &context_;
};

template<typename A>
std::optional<Shape> GetShape(FoldingContext &context, const A &x) {
  return GetShapeHelper{context}.GetShape(x);
}
}
#endif  // FORTRAN_EVALUATE_SHAPE_H_
