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

#include "shape.h"
#include "fold.h"
#include "tools.h"
#include "traversal.h"
#include "../common/idioms.h"
#include "../common/template.h"
#include "../semantics/symbol.h"

namespace Fortran::evaluate {

Shape AsShape(const Constant<ExtentType> &arrayConstant) {
  CHECK(arrayConstant.Rank() == 1);
  Shape result;
  std::size_t dimensions{arrayConstant.size()};
  for (std::size_t j{0}; j < dimensions; ++j) {
    Scalar<ExtentType> extent{arrayConstant.values().at(j)};
    result.emplace_back(MaybeExtent{ExtentExpr{extent}});
  }
  return result;
}

std::optional<Shape> AsShape(ExtentExpr &&arrayExpr) {
  if (auto *constArray{UnwrapExpr<Constant<ExtentType>>(arrayExpr)}) {
    return AsShape(*constArray);
  }
  if (auto *constructor{UnwrapExpr<ArrayConstructor<ExtentType>>(arrayExpr)}) {
    Shape result;
    for (auto &value : constructor->values()) {
      if (auto *expr{std::get_if<ExtentExpr>(&value.u)}) {
        if (expr->Rank() == 0) {
          result.emplace_back(std::move(*expr));
          continue;
        }
      }
      return std::nullopt;
    }
    return result;
  }
  // TODO: linearize other array-valued expressions of known shape, e.g. A+B
  // as well as conversions of arrays; this will be easier given a
  // general-purpose array expression flattener (pmk)
  return std::nullopt;
}

std::optional<ExtentExpr> AsShapeArrayExpr(const Shape &shape) {
  ArrayConstructorValues<ExtentType> values;
  for (const auto &dim : shape) {
    if (dim.has_value()) {
      values.Push(common::Clone(*dim));
    } else {
      return std::nullopt;
    }
  }
  return ExtentExpr{ArrayConstructor<ExtentType>{std::move(values)}};
}

std::optional<Constant<ExtentType>> AsConstantShape(const Shape &shape) {
  if (auto shapeArray{AsShapeArrayExpr(shape)}) {
    FoldingContext noFoldingContext;
    auto folded{Fold(noFoldingContext, std::move(*shapeArray))};
    if (auto *p{UnwrapExpr<Constant<ExtentType>>(folded)}) {
      return std::move(*p);
    }
  }
  return std::nullopt;
}

static ExtentExpr ComputeTripCount(
    ExtentExpr &&lower, ExtentExpr &&upper, ExtentExpr &&stride) {
  ExtentExpr strideCopy{common::Clone(stride)};
  ExtentExpr span{
      (std::move(upper) - std::move(lower) + std::move(strideCopy)) /
      std::move(stride)};
  ExtentExpr extent{
      Extremum<ExtentType>{std::move(span), ExtentExpr{0}, Ordering::Greater}};
  FoldingContext noFoldingContext;
  return Fold(noFoldingContext, std::move(extent));
}

ExtentExpr CountTrips(
    ExtentExpr &&lower, ExtentExpr &&upper, ExtentExpr &&stride) {
  return ComputeTripCount(
      std::move(lower), std::move(upper), std::move(stride));
}

ExtentExpr CountTrips(const ExtentExpr &lower, const ExtentExpr &upper,
    const ExtentExpr &stride) {
  return ComputeTripCount(
      common::Clone(lower), common::Clone(upper), common::Clone(stride));
}

MaybeExtent CountTrips(
    MaybeExtent &&lower, MaybeExtent &&upper, MaybeExtent &&stride) {
  return common::MapOptional(
      ComputeTripCount, std::move(lower), std::move(upper), std::move(stride));
}

MaybeExtent GetSize(Shape &&shape) {
  ExtentExpr extent{1};
  for (auto &&dim : std::move(shape)) {
    if (dim.has_value()) {
      extent = std::move(extent) * std::move(*dim);
    } else {
      return std::nullopt;
    }
  }
  return extent;
}

static MaybeExtent GetLowerBound(
    const Symbol &symbol, const Component *component, int dimension) {
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    int j{0};
    for (const auto &shapeSpec : details->shape()) {
      if (j++ == dimension) {
        if (const auto &bound{shapeSpec.lbound().GetExplicit()}) {
          return *bound;
        } else if (component != nullptr) {
          return ExtentExpr{DescriptorInquiry{
              *component, DescriptorInquiry::Field::LowerBound, dimension}};
        } else {
          return ExtentExpr{DescriptorInquiry{
              symbol, DescriptorInquiry::Field::LowerBound, dimension}};
        }
      }
    }
  }
  return std::nullopt;
}

static MaybeExtent GetExtent(
    const Symbol &symbol, const Component *component, int dimension) {
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    int j{0};
    for (const auto &shapeSpec : details->shape()) {
      if (j++ == dimension) {
        if (const auto &lbound{shapeSpec.lbound().GetExplicit()}) {
          if (const auto &ubound{shapeSpec.ubound().GetExplicit()}) {
            FoldingContext noFoldingContext;
            return Fold(noFoldingContext,
                common::Clone(ubound.value()) - common::Clone(lbound.value()) +
                    ExtentExpr{1});
          }
        }
        if (component != nullptr) {
          return ExtentExpr{DescriptorInquiry{
              *component, DescriptorInquiry::Field::Extent, dimension}};
        } else {
          return ExtentExpr{DescriptorInquiry{
              &symbol, DescriptorInquiry::Field::Extent, dimension}};
        }
      }
    }
  }
  return std::nullopt;
}

static MaybeExtent GetExtent(const Subscript &subscript, const Symbol &symbol,
    const Component *component, int dimension) {
  return std::visit(
      common::visitors{
          [&](const Triplet &triplet) -> MaybeExtent {
            MaybeExtent upper{triplet.upper()};
            if (!upper.has_value()) {
              upper = GetExtent(symbol, component, dimension);
            }
            MaybeExtent lower{triplet.lower()};
            if (!lower.has_value()) {
              lower = GetLowerBound(symbol, component, dimension);
            }
            return CountTrips(std::move(lower), std::move(upper),
                MaybeExtent{triplet.stride()});
          },
          [](const IndirectSubscriptIntegerExpr &subs) -> MaybeExtent {
            if (auto shape{GetShape(subs.value())}) {
              if (shape->size() > 0) {
                CHECK(shape->size() == 1);  // vector-valued subscript
                return std::move(shape->at(0));
              }
            }
            return std::nullopt;
          },
      },
      subscript.u);
}

bool ContainsAnyImpliedDoIndex(const ExtentExpr &expr) {
  struct MyVisitor : public virtual VisitorBase<bool> {
    using Result = bool;
    explicit MyVisitor(int) { result() = false; }
    void Handle(const ImpliedDoIndex &) { Return(true); }
  };
  return Visitor<MyVisitor>{0}.Traverse(expr);
}

std::optional<Shape> GetShape(
    const Symbol &symbol, const Component *component) {
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    Shape result;
    int n = details->shape().size();
    for (int dimension{0}; dimension < n; ++dimension) {
      result.emplace_back(GetExtent(symbol, component, dimension++));
    }
    return result;
  } else {
    return std::nullopt;
  }
}

std::optional<Shape> GetShape(const Symbol *symbol) {
  if (symbol != nullptr) {
    return GetShape(*symbol);
  } else {
    return std::nullopt;
  }
}

std::optional<Shape> GetShape(const BaseObject &object) {
  if (const Symbol * symbol{object.symbol()}) {
    return GetShape(*symbol);
  } else {
    return Shape{};
  }
}

std::optional<Shape> GetShape(const Component &component) {
  const Symbol &symbol{component.GetLastSymbol()};
  if (symbol.Rank() > 0) {
    return GetShape(symbol, &component);
  } else {
    return GetShape(component.base());
  }
}

std::optional<Shape> GetShape(const ArrayRef &arrayRef) {
  Shape shape;
  const Symbol &symbol{arrayRef.GetLastSymbol()};
  const Component *component{std::get_if<Component>(&arrayRef.base())};
  int dimension{0};
  for (const Subscript &ss : arrayRef.subscript()) {
    if (ss.Rank() > 0) {
      shape.emplace_back(GetExtent(ss, symbol, component, dimension));
    }
    ++dimension;
  }
  if (shape.empty()) {
    return GetShape(arrayRef.base());
  } else {
    return shape;
  }
}

std::optional<Shape> GetShape(const CoarrayRef &coarrayRef) {
  Shape shape;
  SymbolOrComponent base{coarrayRef.GetBaseSymbolOrComponent()};
  const Symbol &symbol{coarrayRef.GetLastSymbol()};
  const Component *component{std::get_if<Component>(&base)};
  int dimension{0};
  for (const Subscript &ss : coarrayRef.subscript()) {
    if (ss.Rank() > 0) {
      shape.emplace_back(GetExtent(ss, symbol, component, dimension));
    }
    ++dimension;
  }
  if (shape.empty()) {
    return GetShape(coarrayRef.GetLastSymbol());
  } else {
    return shape;
  }
}

std::optional<Shape> GetShape(const DataRef &dataRef) {
  return GetShape(dataRef.u);
}

std::optional<Shape> GetShape(const Substring &substring) {
  if (const auto *dataRef{substring.GetParentIf<DataRef>()}) {
    return GetShape(*dataRef);
  } else {
    return std::nullopt;
  }
}

std::optional<Shape> GetShape(const ComplexPart &part) {
  return GetShape(part.complex());
}

std::optional<Shape> GetShape(const ActualArgument &arg) {
  return GetShape(arg.value());
}

std::optional<Shape> GetShape(const ProcedureRef &call) {
  if (call.Rank() == 0) {
    return Shape{};
  } else if (call.IsElemental()) {
    for (const auto &arg : call.arguments()) {
      if (arg.has_value() && arg->Rank() > 0) {
        return GetShape(*arg);
      }
    }
  } else if (const Symbol * symbol{call.proc().GetSymbol()}) {
    return GetShape(*symbol);
  } else if (const auto *intrinsic{
                 std::get_if<SpecificIntrinsic>(&call.proc().u)}) {
    if (intrinsic->name == "shape" || intrinsic->name == "lbound" ||
        intrinsic->name == "ubound") {
      return Shape{MaybeExtent{
          ExtentExpr{call.arguments().front().value().value().Rank()}}};
    } else if (intrinsic->name == "reshape") {
      if (call.arguments().size() >= 2 && call.arguments().at(1).has_value()) {
        // SHAPE(RESHAPE(array,shape)) -> shape
        const Expr<SomeType> &shapeExpr{call.arguments().at(1)->value()};
        Expr<SomeInteger> shape{std::get<Expr<SomeInteger>>(shapeExpr.u)};
        return AsShape(ConvertToType<ExtentType>(std::move(shape)));
      }
    } else {
      // TODO: shapes of other non-elemental intrinsic results
    }
  }
  return std::nullopt;
}

std::optional<Shape> GetShape(const Relational<SomeType> &relation) {
  return GetShape(relation.u);
}

std::optional<Shape> GetShape(const StructureConstructor &) {
  return Shape{};  // always scalar
}

std::optional<Shape> GetShape(const ImpliedDoIndex &) {
  return Shape{};  // always scalar
}

std::optional<Shape> GetShape(const DescriptorInquiry &) {
  return Shape{};  // always scalar
}

std::optional<Shape> GetShape(const BOZLiteralConstant &) {
  return Shape{};  // always scalar
}

std::optional<Shape> GetShape(const NullPointer &) {
  return {};  // not an object
}

}
