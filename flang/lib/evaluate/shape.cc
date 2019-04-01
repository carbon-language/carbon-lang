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
std::optional<Shape> GetShape(
    const semantics::Symbol &symbol, const Component *component) {
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    Shape result;
    int dimension{1};
    for (const auto &shapeSpec : details->shape()) {
      if (shapeSpec.isExplicit()) {
        result.emplace_back(
            common::Clone(shapeSpec.ubound().GetExplicit().value()) -
            common::Clone(shapeSpec.lbound().GetExplicit().value()) +
            Expr<SubscriptInteger>{1});
      } else if (component != nullptr) {
        result.emplace_back(Expr<SubscriptInteger>{DescriptorInquiry{
            *component, DescriptorInquiry::Field::Extent, dimension}});
      } else {
        result.emplace_back(Expr<SubscriptInteger>{DescriptorInquiry{
            symbol, DescriptorInquiry::Field::Extent, dimension}});
      }
      ++dimension;
    }
    return result;
  } else {
    return std::nullopt;
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
static Extent GetExtent(const Subscript &subscript) {
  return std::visit(
      common::visitors{
          [](const Triplet &triplet) -> Extent {
            if (auto lower{triplet.lower()}) {
              if (auto lowerValue{ToInt64(*lower)}) {
                if (auto upper{triplet.upper()}) {
                  if (auto upperValue{ToInt64(*upper)}) {
                    if (auto strideValue{ToInt64(triplet.stride())}) {
                      if (*strideValue != 0) {
                        std::int64_t extent{
                            (*upperValue - *lowerValue + *strideValue) /
                            *strideValue};
                        return Expr<SubscriptInteger>{extent > 0 ? extent : 0};
                      }
                    }
                  }
                }
              }
            }
            return std::nullopt;
          },
          [](const IndirectSubscriptIntegerExpr &subs) -> Extent {
            if (auto shape{GetShape(subs.value())}) {
              if (shape->size() == 1) {
                return std::move(shape->at(0));
              }
            }
            return std::nullopt;
          },
      },
      subscript.u);
}
std::optional<Shape> GetShape(const ArrayRef &arrayRef) {
  int subscripts{arrayRef.size()};
  Shape shape;
  for (int j = 0; j < subscripts; ++j) {
    const Subscript &subscript{arrayRef.at(j)};
    if (subscript.Rank() > 0) {
      shape.emplace_back(GetExtent(subscript));
    }
  }
  if (shape.empty()) {
    return GetShape(arrayRef.base());
  } else {
    return shape;
  }
}
std::optional<Shape> GetShape(const CoarrayRef &);  // TODO pmk
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
}
