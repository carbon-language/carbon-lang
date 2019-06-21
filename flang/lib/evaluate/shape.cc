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
#include "../parser/message.h"
#include "../semantics/symbol.h"

namespace Fortran::evaluate {

bool IsImpliedShape(const Symbol &symbol) {
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    if (symbol.attrs().test(semantics::Attr::PARAMETER) &&
        details->init().has_value()) {
      for (const semantics::ShapeSpec &ss : details->shape()) {
        if (!ss.ubound().isDeferred()) {
          // ss.isDeferred() can't be used because the lower bounds are
          // implicitly set to 1 in the symbol table.
          return false;
        }
      }
      return !details->shape().empty();
    }
  }
  return false;
}

bool IsExplicitShape(const Symbol &symbol) {
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    for (const semantics::ShapeSpec &ss : details->shape()) {
      if (!ss.isExplicit()) {
        return false;
      }
    }
    return true;  // even if scalar
  } else {
    return false;
  }
}

Shape AsShape(const Constant<ExtentType> &arrayConstant) {
  CHECK(arrayConstant.Rank() == 1);
  Shape result;
  std::size_t dimensions{arrayConstant.size()};
  for (std::size_t j{0}; j < dimensions; ++j) {
    Scalar<ExtentType> extent{arrayConstant.values().at(j)};
    result.emplace_back(MaybeExtentExpr{ExtentExpr{extent}});
  }
  return result;
}

std::optional<Shape> AsShape(FoldingContext &context, ExtentExpr &&arrayExpr) {
  // Flatten any array expression into an array constructor if possible.
  arrayExpr = Fold(context, std::move(arrayExpr));
  if (const auto *constArray{UnwrapConstantValue<ExtentType>(arrayExpr)}) {
    return AsShape(*constArray);
  }
  if (auto *constructor{UnwrapExpr<ArrayConstructor<ExtentType>>(arrayExpr)}) {
    Shape result;
    for (auto &value : *constructor) {
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
  return std::nullopt;
}

std::optional<ExtentExpr> AsExtentArrayExpr(const Shape &shape) {
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
  if (auto shapeArray{AsExtentArrayExpr(shape)}) {
    FoldingContext noFoldingContext;
    auto folded{Fold(noFoldingContext, std::move(*shapeArray))};
    if (auto *p{UnwrapConstantValue<ExtentType>(folded)}) {
      return std::move(*p);
    }
  }
  return std::nullopt;
}

Constant<SubscriptInteger> AsConstantShape(const ConstantSubscripts &shape) {
  using IntType = Scalar<SubscriptInteger>;
  std::vector<IntType> result;
  for (auto dim : shape) {
    result.emplace_back(dim);
  }
  return {std::move(result), ConstantSubscripts{GetRank(shape)}};
}

ConstantSubscripts AsConstantExtents(const Constant<ExtentType> &shape) {
  ConstantSubscripts result;
  for (const auto &extent : shape.values()) {
    result.push_back(extent.ToInt64());
  }
  return result;
}

std::optional<ConstantSubscripts> AsConstantExtents(const Shape &shape) {
  if (auto shapeConstant{AsConstantShape(shape)}) {
    return AsConstantExtents(*shapeConstant);
  } else {
    return std::nullopt;
  }
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

MaybeExtentExpr CountTrips(MaybeExtentExpr &&lower, MaybeExtentExpr &&upper,
    MaybeExtentExpr &&stride) {
  return common::MapOptional(
      ComputeTripCount, std::move(lower), std::move(upper), std::move(stride));
}

MaybeExtentExpr GetSize(Shape &&shape) {
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

bool ContainsAnyImpliedDoIndex(const ExtentExpr &expr) {
  struct MyVisitor : public virtual VisitorBase<bool> {
    using Result = bool;
    explicit MyVisitor(int) { result() = false; }
    void Handle(const ImpliedDoIndex &) { Return(true); }
  };
  return Visitor<MyVisitor>{0}.Traverse(expr);
}

MaybeExtentExpr GetLowerBound(FoldingContext &context, const Symbol &symbol,
    int dimension, const Component *component) {
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    int j{0};
    for (const auto &shapeSpec : details->shape()) {
      if (j++ == dimension) {
        if (const auto &bound{shapeSpec.lbound().GetExplicit()}) {
          return Fold(context, common::Clone(*bound));
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

Shape GetLowerBounds(
    FoldingContext &context, const Symbol &symbol, const Component *component) {
  Shape result;
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    int dim{0};
    for (const auto &shapeSpec : details->shape()) {
      if (const auto &bound{shapeSpec.lbound().GetExplicit()}) {
        result.emplace_back(Fold(context, common::Clone(*bound)));
      } else if (component != nullptr) {
        result.emplace_back(ExtentExpr{DescriptorInquiry{
            *component, DescriptorInquiry::Field::LowerBound, dim}});
      } else {
        result.emplace_back(ExtentExpr{DescriptorInquiry{
            symbol, DescriptorInquiry::Field::LowerBound, dim}});
      }
      ++dim;
    }
  }
  return result;
}

MaybeExtentExpr GetExtent(FoldingContext &context, const Symbol &symbol,
    int dimension, const Component *component) {
  CHECK(dimension >= 0);
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    if (IsImpliedShape(symbol)) {
      Shape shape{GetShape(context, symbol).value()};
      return std::move(shape.at(dimension));
    }
    int j{0};
    for (const auto &shapeSpec : details->shape()) {
      if (j++ == dimension) {
        if (shapeSpec.isExplicit()) {
          if (const auto &ubound{shapeSpec.ubound().GetExplicit()}) {
            if (const auto &lbound{shapeSpec.lbound().GetExplicit()}) {
              return Fold(context,
                  common::Clone(ubound.value()) -
                      common::Clone(lbound.value()) + ExtentExpr{1});
            } else {
              return Fold(context, common::Clone(ubound.value()));
            }
          }
        } else if (details->IsAssumedSize() && j == symbol.Rank()) {
          return std::nullopt;
        } else if (component != nullptr) {
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

MaybeExtentExpr GetExtent(FoldingContext &context, const Subscript &subscript,
    const Symbol &symbol, int dimension, const Component *component) {
  return std::visit(
      common::visitors{
          [&](const Triplet &triplet) -> MaybeExtentExpr {
            MaybeExtentExpr upper{triplet.upper()};
            if (!upper.has_value()) {
              upper = GetExtent(context, symbol, dimension, component);
            }
            MaybeExtentExpr lower{triplet.lower()};
            if (!lower.has_value()) {
              lower = GetLowerBound(context, symbol, dimension, component);
            }
            return CountTrips(std::move(lower), std::move(upper),
                MaybeExtentExpr{triplet.stride()});
          },
          [&](const IndirectSubscriptIntegerExpr &subs) -> MaybeExtentExpr {
            if (auto shape{GetShape(context, subs.value())}) {
              if (GetRank(*shape) > 0) {
                CHECK(GetRank(*shape) == 1);  // vector-valued subscript
                return std::move(shape->at(0));
              }
            }
            return std::nullopt;
          },
      },
      subscript.u);
}

MaybeExtentExpr GetUpperBound(FoldingContext &context, MaybeExtentExpr &&lower,
    MaybeExtentExpr &&extent) {
  if (lower.has_value() && extent.has_value()) {
    return Fold(
        context, std::move(*extent) - std::move(*lower) + ExtentExpr{1});
  } else {
    return std::nullopt;
  }
}

std::optional<Shape> GetShapeHelper::GetShape(
    const Symbol &symbol, const Component *component) {
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    if (IsImpliedShape(symbol)) {
      return GetShape(*details->init());
    } else {
      Shape result;
      int n{static_cast<int>(details->shape().size())};
      for (int dimension{0}; dimension < n; ++dimension) {
        result.emplace_back(GetExtent(context_, symbol, dimension, component));
      }
      return result;
    }
  } else if (const auto *details{
                 symbol.detailsIf<semantics::AssocEntityDetails>()}) {
    if (details->expr().has_value()) {
      return GetShape(*details->expr());
    }
  }
  return std::nullopt;
}

std::optional<Shape> GetShapeHelper::GetShape(const Symbol *symbol) {
  if (symbol != nullptr) {
    return GetShape(*symbol);
  } else {
    return std::nullopt;
  }
}

std::optional<Shape> GetShapeHelper::GetShape(const BaseObject &object) {
  if (const Symbol * symbol{object.symbol()}) {
    return GetShape(*symbol);
  } else {
    return Shape{};
  }
}

std::optional<Shape> GetShapeHelper::GetShape(const Component &component) {
  const Symbol &symbol{component.GetLastSymbol()};
  if (symbol.Rank() > 0) {
    return GetShape(symbol, &component);
  } else {
    return GetShape(component.base());
  }
}

std::optional<Shape> GetShapeHelper::GetShape(const ArrayRef &arrayRef) {
  Shape shape;
  const Symbol &symbol{arrayRef.GetLastSymbol()};
  const Component *component{std::get_if<Component>(&arrayRef.base())};
  int dimension{0};
  for (const Subscript &ss : arrayRef.subscript()) {
    if (ss.Rank() > 0) {
      shape.emplace_back(GetExtent(context_, ss, symbol, dimension, component));
    }
    ++dimension;
  }
  if (shape.empty()) {
    return GetShape(arrayRef.base());
  } else {
    return shape;
  }
}

std::optional<Shape> GetShapeHelper::GetShape(const CoarrayRef &coarrayRef) {
  Shape shape;
  SymbolOrComponent base{coarrayRef.GetBaseSymbolOrComponent()};
  const Symbol &symbol{coarrayRef.GetLastSymbol()};
  const Component *component{std::get_if<Component>(&base)};
  int dimension{0};
  for (const Subscript &ss : coarrayRef.subscript()) {
    if (ss.Rank() > 0) {
      shape.emplace_back(GetExtent(context_, ss, symbol, dimension, component));
    }
    ++dimension;
  }
  if (shape.empty()) {
    return GetShape(coarrayRef.GetLastSymbol());
  } else {
    return shape;
  }
}

std::optional<Shape> GetShapeHelper::GetShape(const DataRef &dataRef) {
  return GetShape(dataRef.u);
}

std::optional<Shape> GetShapeHelper::GetShape(const Substring &substring) {
  if (const auto *dataRef{substring.GetParentIf<DataRef>()}) {
    return GetShape(*dataRef);
  } else {
    return std::nullopt;
  }
}

std::optional<Shape> GetShapeHelper::GetShape(const ComplexPart &part) {
  return GetShape(part.complex());
}

std::optional<Shape> GetShapeHelper::GetShape(const ActualArgument &arg) {
  if (const auto *expr{arg.UnwrapExpr()}) {
    return GetShape(*expr);
  } else {
    const Symbol *aType{arg.GetAssumedTypeDummy()};
    CHECK(aType != nullptr);
    return GetShape(*aType);
  }
}

std::optional<Shape> GetShapeHelper::GetShape(const ProcedureDesignator &proc) {
  if (const Symbol * symbol{proc.GetSymbol()}) {
    return GetShape(*symbol);
  } else {
    return std::nullopt;
  }
}

std::optional<Shape> GetShapeHelper::GetShape(const ProcedureRef &call) {
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
      const auto *expr{call.arguments().front().value().UnwrapExpr()};
      CHECK(expr != nullptr);
      return Shape{MaybeExtentExpr{ExtentExpr{expr->Rank()}}};
    } else if (intrinsic->name == "reshape") {
      if (call.arguments().size() >= 2 && call.arguments().at(1).has_value()) {
        // SHAPE(RESHAPE(array,shape)) -> shape
        const auto *shapeExpr{call.arguments().at(1).value().UnwrapExpr()};
        CHECK(shapeExpr != nullptr);
        Expr<SomeInteger> shape{std::get<Expr<SomeInteger>>(shapeExpr->u)};
        return AsShape(context_, ConvertToType<ExtentType>(std::move(shape)));
      }
    } else {
      // TODO: shapes of other non-elemental intrinsic results
    }
  }
  return std::nullopt;
}

std::optional<Shape> GetShapeHelper::GetShape(
    const Relational<SomeType> &relation) {
  return GetShape(relation.u);
}

std::optional<Shape> GetShapeHelper::GetShape(const StructureConstructor &) {
  return Shape{};  // always scalar
}

std::optional<Shape> GetShapeHelper::GetShape(const ImpliedDoIndex &) {
  return Shape{};  // always scalar
}

std::optional<Shape> GetShapeHelper::GetShape(const DescriptorInquiry &) {
  return Shape{};  // always scalar
}

std::optional<Shape> GetShapeHelper::GetShape(const BOZLiteralConstant &) {
  return Shape{};  // always scalar
}

std::optional<Shape> GetShapeHelper::GetShape(const NullPointer &) {
  return {};  // not an object
}

bool CheckConformance(parser::ContextualMessages &messages, const Shape &left,
    const Shape &right, const char *leftDesc, const char *rightDesc) {
  if (!left.empty() && !right.empty()) {
    int n{GetRank(left)};
    int rn{GetRank(right)};
    if (n != rn) {
      messages.Say("Rank of %s is %d, but %s has rank %d"_err_en_US, leftDesc,
          n, rightDesc, rn);
      return false;
    } else {
      for (int j{0}; j < n; ++j) {
        if (auto leftDim{ToInt64(left[j])}) {
          if (auto rightDim{ToInt64(right[j])}) {
            if (*leftDim != *rightDim) {
              messages.Say("Dimension %d of %s has extent %jd, "
                           "but %s has extent %jd"_err_en_US,
                  j + 1, leftDesc, static_cast<std::intmax_t>(*leftDim),
                  rightDesc, static_cast<std::intmax_t>(*rightDim));
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

}
