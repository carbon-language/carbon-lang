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
#include "type.h"
#include "../common/idioms.h"
#include "../common/template.h"
#include "../parser/message.h"
#include "../semantics/symbol.h"

namespace Fortran::evaluate {

bool IsImpliedShape(const Symbol &symbol0) {
  const Symbol &symbol{ResolveAssociations(symbol0)};
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

bool IsExplicitShape(const Symbol &symbol0) {
  const Symbol &symbol{ResolveAssociations(symbol0)};
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
    common::IntrinsicTypeDefaultKinds defaults;
    FoldingContext noFoldingContext{defaults};
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
  common::IntrinsicTypeDefaultKinds defaults;
  FoldingContext noFoldingContext{defaults};
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

ExtentExpr GetLowerBound(
    FoldingContext &context, const NamedEntity &base, int dimension) {
  const Symbol &symbol{ResolveAssociations(base.GetLastSymbol())};
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    int j{0};
    for (const auto &shapeSpec : details->shape()) {
      if (j++ == dimension) {
        if (const auto &bound{shapeSpec.lbound().GetExplicit()}) {
          return Fold(context, common::Clone(*bound));
        } else if (semantics::IsDescriptor(symbol)) {
          return ExtentExpr{DescriptorInquiry{
              base, DescriptorInquiry::Field::LowerBound, dimension}};
        } else {
          break;
        }
      }
    }
  }
  // When we don't know that we don't know the lower bound at compilation
  // time, then we do know it, and it's one.  (See LBOUND, 16.9.109).
  return ExtentExpr{1};
}

Shape GetLowerBounds(FoldingContext &context, const NamedEntity &base) {
  const Symbol &symbol{ResolveAssociations(base.GetLastSymbol())};
  Shape result;
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    int dim{0};
    for (const auto &shapeSpec : details->shape()) {
      if (const auto &bound{shapeSpec.lbound().GetExplicit()}) {
        result.emplace_back(Fold(context, common::Clone(*bound)));
      } else if (semantics::IsDescriptor(symbol)) {
        result.emplace_back(ExtentExpr{DescriptorInquiry{
            base, DescriptorInquiry::Field::LowerBound, dim}});
      } else {
        result.emplace_back(std::nullopt);
      }
      ++dim;
    }
  } else {
    int rank{base.Rank()};
    for (int dim{0}; dim < rank; ++dim) {
      result.emplace_back(ExtentExpr{1});
    }
  }
  CHECK(GetRank(result) == symbol.Rank());
  return result;
}

MaybeExtentExpr GetExtent(
    FoldingContext &context, const NamedEntity &base, int dimension) {
  CHECK(dimension >= 0);
  const Symbol &symbol{ResolveAssociations(base.GetLastSymbol())};
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
        } else if (semantics::IsDescriptor(symbol)) {
          return ExtentExpr{DescriptorInquiry{
              NamedEntity{base}, DescriptorInquiry::Field::Extent, dimension}};
        }
      }
    }
  }
  return std::nullopt;
}

MaybeExtentExpr GetExtent(FoldingContext &context, const Subscript &subscript,
    const NamedEntity &base, int dimension) {
  return std::visit(
      common::visitors{
          [&](const Triplet &triplet) -> MaybeExtentExpr {
            MaybeExtentExpr upper{triplet.upper()};
            if (!upper.has_value()) {
              upper = GetUpperBound(context, base, dimension);
            }
            MaybeExtentExpr lower{triplet.lower()};
            if (!lower.has_value()) {
              lower = GetLowerBound(context, base, dimension);
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

MaybeExtentExpr ComputeUpperBound(
    FoldingContext &context, ExtentExpr &&lower, MaybeExtentExpr &&extent) {
  if (extent.has_value()) {
    return Fold(context, std::move(*extent) - std::move(lower) + ExtentExpr{1});
  } else {
    return std::nullopt;
  }
}

MaybeExtentExpr GetUpperBound(
    FoldingContext &context, const NamedEntity &base, int dimension) {
  const Symbol &symbol{ResolveAssociations(base.GetLastSymbol())};
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    int j{0};
    for (const auto &shapeSpec : details->shape()) {
      if (j++ == dimension) {
        if (const auto &bound{shapeSpec.ubound().GetExplicit()}) {
          return Fold(context, common::Clone(*bound));
        } else if (details->IsAssumedSize() && dimension + 1 == symbol.Rank()) {
          break;
        } else {
          return ComputeUpperBound(context,
              GetLowerBound(context, base, dimension),
              GetExtent(context, base, dimension));
        }
      }
    }
  }
  return std::nullopt;
}

Shape GetUpperBounds(FoldingContext &context, const NamedEntity &base) {
  const Symbol &symbol{ResolveAssociations(base.GetLastSymbol())};
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    Shape result;
    int dim{0};
    for (const auto &shapeSpec : details->shape()) {
      if (const auto &bound{shapeSpec.ubound().GetExplicit()}) {
        result.emplace_back(Fold(context, common::Clone(*bound)));
      } else if (details->IsAssumedSize()) {
        CHECK(dim + 1 == base.Rank());
        result.emplace_back(std::nullopt);  // UBOUND folding replaces with -1
      } else {
        result.emplace_back(ComputeUpperBound(context,
            GetLowerBound(context, base, dim), GetExtent(context, base, dim)));
      }
      ++dim;
    }
    CHECK(GetRank(result) == symbol.Rank());
    return result;
  } else {
    return std::move(GetShape(context, base).value());
  }
}

void GetShapeVisitor::Handle(const Symbol &symbol) {
  std::visit(
      common::visitors{
          [&](const semantics::ObjectEntityDetails &object) {
            Handle(NamedEntity{symbol});
          },
          [&](const semantics::AssocEntityDetails &assoc) {
            Nested(assoc.expr());
          },
          [&](const semantics::SubprogramDetails &subp) {
            if (subp.isFunction()) {
              Handle(subp.result());
            } else {
              Return();
            }
          },
          [&](const semantics::ProcBindingDetails &binding) {
            Handle(binding.symbol());
          },
          [&](const semantics::UseDetails &use) { Handle(use.symbol()); },
          [&](const semantics::HostAssocDetails &assoc) {
            Handle(assoc.symbol());
          },
          [&](const auto &) { Return(); },
      },
      symbol.details());
}

void GetShapeVisitor::Handle(const Component &component) {
  if (component.GetLastSymbol().Rank() > 0) {
    Handle(NamedEntity{Component{component}});
  } else {
    Nested(component.base());
  }
}

void GetShapeVisitor::Handle(const NamedEntity &base) {
  const Symbol &symbol{base.GetLastSymbol()};
  if (const auto *object{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    if (IsImpliedShape(symbol)) {
      Nested(object->init());
    } else {
      Shape result;
      int n{static_cast<int>(object->shape().size())};
      for (int dimension{0}; dimension < n; ++dimension) {
        result.emplace_back(GetExtent(context_, base, dimension));
      }
      Return(std::move(result));
    }
  } else {
    Handle(symbol);
  }
}

void GetShapeVisitor::Handle(const Substring &substring) {
  Nested(substring.parent());
}

void GetShapeVisitor::Handle(const ArrayRef &arrayRef) {
  Shape shape;
  int dimension{0};
  for (const Subscript &ss : arrayRef.subscript()) {
    if (ss.Rank() > 0) {
      shape.emplace_back(GetExtent(context_, ss, arrayRef.base(), dimension));
    }
    ++dimension;
  }
  if (shape.empty()) {
    Nested(arrayRef.base());
  } else {
    Return(std::move(shape));
  }
}

void GetShapeVisitor::Handle(const CoarrayRef &coarrayRef) {
  Shape shape;
  NamedEntity base{coarrayRef.GetBase()};
  int dimension{0};
  for (const Subscript &ss : coarrayRef.subscript()) {
    if (ss.Rank() > 0) {
      shape.emplace_back(GetExtent(context_, ss, base, dimension));
    }
    ++dimension;
  }
  if (shape.empty()) {
    Nested(base);
  } else {
    Return(std::move(shape));
  }
}

void GetShapeVisitor::Handle(const ProcedureRef &call) {
  if (call.Rank() == 0) {
    Scalar();
  } else if (call.IsElemental()) {
    for (const auto &arg : call.arguments()) {
      if (arg.has_value() && arg->Rank() > 0) {
        Nested(*arg);
        return;
      }
    }
    Scalar();
  } else if (const Symbol * symbol{call.proc().GetSymbol()}) {
    Handle(*symbol);
  } else if (const auto *intrinsic{
                 std::get_if<SpecificIntrinsic>(&call.proc().u)}) {
    if (intrinsic->name == "shape" || intrinsic->name == "lbound" ||
        intrinsic->name == "ubound") {
      const auto *expr{call.arguments().front().value().UnwrapExpr()};
      CHECK(expr != nullptr);
      Return(Shape{MaybeExtentExpr{ExtentExpr{expr->Rank()}}});
    } else if (intrinsic->name == "reshape") {
      if (call.arguments().size() >= 2 && call.arguments().at(1).has_value()) {
        // SHAPE(RESHAPE(array,shape)) -> shape
        const auto *shapeExpr{call.arguments().at(1).value().UnwrapExpr()};
        CHECK(shapeExpr != nullptr);
        Expr<SomeInteger> shape{std::get<Expr<SomeInteger>>(shapeExpr->u)};
        Return(AsShape(context_, ConvertToType<ExtentType>(std::move(shape))));
      }
    } else {
      // TODO: shapes of other non-elemental intrinsic results
    }
  }
  Return();
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
