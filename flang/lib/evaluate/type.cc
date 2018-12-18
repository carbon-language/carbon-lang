// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

#include "type.h"
#include "../common/idioms.h"
#include "../semantics/symbol.h"
#include "../semantics/type.h"
#include <algorithm>
#include <optional>
#include <string>

using namespace std::literals::string_literals;

namespace Fortran::evaluate {

std::optional<DynamicType> GetSymbolType(const semantics::Symbol &symbol) {
  if (const auto *type{symbol.GetType()}) {
    if (const auto *intrinsic{type->AsIntrinsic()}) {
      TypeCategory category{intrinsic->category()};
      int kind{intrinsic->kind()};
      if (IsValidKindOfIntrinsicType(category, kind)) {
        return DynamicType{category, kind};
      }
    } else if (const auto *derived{type->AsDerived()}) {
      return DynamicType{TypeCategory::Derived, 0, derived};
    }
  }
  return std::nullopt;
}

std::string DynamicType::AsFortran() const {
  if (category == TypeCategory::Derived) {
    // TODO: derived type parameters
    return "TYPE("s + derived->name().ToString() + ')';
  } else {
    // TODO: CHARACTER length
    return EnumToString(category) + '(' + std::to_string(kind) + ')';
  }
}

DynamicType DynamicType::ResultTypeForMultiply(const DynamicType &that) const {
  switch (category) {
  case TypeCategory::Integer:
    switch (that.category) {
    case TypeCategory::Integer:
      return DynamicType{TypeCategory::Integer, std::max(kind, that.kind)};
    case TypeCategory::Real:
    case TypeCategory::Complex: return that;
    default: CRASH_NO_CASE;
    }
    break;
  case TypeCategory::Real:
    switch (that.category) {
    case TypeCategory::Integer: return *this;
    case TypeCategory::Real:
      return DynamicType{TypeCategory::Real, std::max(kind, that.kind)};
    case TypeCategory::Complex:
      return DynamicType{TypeCategory::Complex, std::max(kind, that.kind)};
    default: CRASH_NO_CASE;
    }
    break;
  case TypeCategory::Complex:
    switch (that.category) {
    case TypeCategory::Integer: return *this;
    case TypeCategory::Real:
    case TypeCategory::Complex:
      return DynamicType{TypeCategory::Complex, std::max(kind, that.kind)};
    default: CRASH_NO_CASE;
    }
    break;
  case TypeCategory::Logical:
    switch (that.category) {
    case TypeCategory::Logical:
      return DynamicType{TypeCategory::Logical, std::max(kind, that.kind)};
    default: CRASH_NO_CASE;
    }
    break;
  default: CRASH_NO_CASE;
  }
  return *this;
}

std::string SomeDerived::AsFortran() const {
  return "TYPE("s + spec().name().ToString() + ')';
}
}
