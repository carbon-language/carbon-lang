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
#include "../common/idioms.h"
#include "../semantics/symbol.h"

namespace Fortran::evaluate {

static Extent GetLowerBound(const semantics::Symbol &symbol,
    const Component *component, int dimension) {
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    int j{0};
    for (const auto &shapeSpec : details->shape()) {
      if (j++ == dimension) {
        if (const auto &bound{shapeSpec.lbound().GetExplicit()}) {
          return *bound;
        } else if (component != nullptr) {
          return Expr<SubscriptInteger>{DescriptorInquiry{
              *component, DescriptorInquiry::Field::LowerBound, dimension}};
        } else {
          return Expr<SubscriptInteger>{DescriptorInquiry{
              symbol, DescriptorInquiry::Field::LowerBound, dimension}};
        }
      }
    }
  }
  return std::nullopt;
}

static Extent GetExtent(const semantics::Symbol &symbol,
    const Component *component, int dimension) {
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    int j{0};
    for (const auto &shapeSpec : details->shape()) {
      if (j++ == dimension) {
        if (const auto &lbound{shapeSpec.lbound().GetExplicit()}) {
          if (const auto &ubound{shapeSpec.ubound().GetExplicit()}) {
            FoldingContext noFoldingContext;
            return Fold(noFoldingContext,
                common::Clone(ubound.value()) - common::Clone(lbound.value()) +
                    Expr<SubscriptInteger>{1});
          }
        }
        if (component != nullptr) {
          return Expr<SubscriptInteger>{DescriptorInquiry{
              *component, DescriptorInquiry::Field::Extent, dimension}};
        } else {
          return Expr<SubscriptInteger>{DescriptorInquiry{
              &symbol, DescriptorInquiry::Field::Extent, dimension}};
        }
      }
    }
  }
  return std::nullopt;
}

static Extent GetExtent(const Subscript &subscript, const Symbol &symbol,
    const Component *component, int dimension) {
  return std::visit(
      common::visitors{
          [&](const Triplet &triplet) -> Extent {
            Extent upper{triplet.upper()};
            if (!upper.has_value()) {
              upper = GetExtent(symbol, component, dimension);
            }
            if (upper.has_value()) {
              Extent lower{triplet.lower()};
              if (!lower.has_value()) {
                lower = GetLowerBound(symbol, component, dimension);
              }
              if (lower.has_value()) {
                auto span{
                    (std::move(*upper) - std::move(*lower) + triplet.stride()) /
                    triplet.stride()};
                Expr<SubscriptInteger> extent{
                    Extremum<SubscriptInteger>{std::move(span),
                        Expr<SubscriptInteger>{0}, Ordering::Greater}};
                FoldingContext noFoldingContext;
                return Fold(noFoldingContext, std::move(extent));
              }
            }
            return std::nullopt;
          },
          [](const IndirectSubscriptIntegerExpr &subs) -> Extent {
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

std::optional<Shape> GetShape(
    const semantics::Symbol &symbol, const Component *component) {
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
  return std::visit([](const auto &x) { return GetShape(x); }, dataRef.u);
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
      return Shape{Extent{Expr<SubscriptInteger>{
          call.arguments().front().value().value().Rank()}}};
    }
    // TODO: shapes of other non-elemental intrinsic results
    // esp. reshape, where shape is value of second argument
  }
  return std::nullopt;
}

std::optional<Shape> GetShape(const StructureConstructor &) {
  return Shape{};  // always scalar
}

std::optional<Shape> GetShape(const BOZLiteralConstant &) {
  return Shape{};  // always scalar
}

std::optional<Shape> GetShape(const NullPointer &) {
  return {};  // not an object
}

}
