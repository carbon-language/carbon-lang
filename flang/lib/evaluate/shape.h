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
#include "traversal.h"
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

template<typename A> std::optional<Shape> GetShape(FoldingContext &, const A &);

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
ExtentExpr GetLowerBound(FoldingContext &, const NamedEntity &, int dimension);
MaybeExtentExpr GetUpperBound(
    FoldingContext &, const NamedEntity &, int dimension);
MaybeExtentExpr ComputeUpperBound(
    FoldingContext &, ExtentExpr &&lower, MaybeExtentExpr &&extent);
Shape GetLowerBounds(FoldingContext &, const NamedEntity &);
Shape GetUpperBounds(FoldingContext &, const NamedEntity &);
MaybeExtentExpr GetExtent(FoldingContext &, const NamedEntity &, int dimension);
MaybeExtentExpr GetExtent(
    FoldingContext &, const Subscript &, const NamedEntity &, int dimension);

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

class GetShapeVisitor : public virtual VisitorBase<std::optional<Shape>> {
public:
  using Result = std::optional<Shape>;
  explicit GetShapeVisitor(FoldingContext &c) : context_{c} {}

  template<typename T> void Handle(const Constant<T> &c) {
    Return(AsShape(c.SHAPE()));
  }
  void Handle(const Symbol &);
  void Handle(const Component &);
  void Handle(const NamedEntity &);
  void Handle(const StaticDataObject::Pointer &) { Scalar(); }
  void Handle(const ArrayRef &);
  void Handle(const CoarrayRef &);
  void Handle(const Substring &);
  void Handle(const ProcedureRef &);
  void Handle(const StructureConstructor &) { Scalar(); }
  template<typename T> void Handle(const ArrayConstructor<T> &aconst) {
    Return(Shape{GetArrayConstructorExtent(aconst)});
  }
  void Handle(const ImpliedDoIndex &) { Scalar(); }
  void Handle(const DescriptorInquiry &) { Scalar(); }
  template<int KIND> void Handle(const TypeParamInquiry<KIND> &) { Scalar(); }
  void Handle(const BOZLiteralConstant &) { Scalar(); }
  void Handle(const NullPointer &) { Return(); }
  template<typename D, typename R, typename LO, typename RO>
  void Handle(const Operation<D, R, LO, RO> &operation) {
    if (operation.right().Rank() > 0) {
      Nested(operation.right());
    } else {
      Nested(operation.left());
    }
  }

private:
  void Scalar() { Return(Shape{}); }

  template<typename A> void Nested(const A &x) {
    Return(GetShape(context_, x));
  }

  template<typename T>
  MaybeExtentExpr GetArrayConstructorValueExtent(
      const ArrayConstructorValue<T> &value) {
    return std::visit(
        common::visitors{
            [&](const Expr<T> &x) -> MaybeExtentExpr {
              if (std::optional<Shape> xShape{GetShape(context_, x)}) {
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
  return Visitor<GetShapeVisitor>{context}.Traverse(x);
}
}
#endif  // FORTRAN_EVALUATE_SHAPE_H_
