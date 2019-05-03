// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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
#include "expression.h"
#include "fold.h"
#include "../common/idioms.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include "../semantics/tools.h"
#include "../semantics/type.h"
#include <algorithm>
#include <optional>
#include <sstream>
#include <string>

using namespace std::literals::string_literals;

// IsDescriptor() predicate
// TODO there's probably a better place for this predicate than here
namespace Fortran::semantics {
static bool IsDescriptor(const ObjectEntityDetails &details) {
  if (const auto *type{details.type()}) {
    if (const IntrinsicTypeSpec * typeSpec{type->AsIntrinsic()}) {
      if (typeSpec->category() == TypeCategory::Character) {
        // TODO maybe character lengths won't be in descriptors
        return true;
      }
    } else if (const DerivedTypeSpec * typeSpec{type->AsDerived()}) {
      if (details.isDummy()) {
        return true;
      }
      // Any length type parameter?
      if (const Scope * scope{typeSpec->scope()}) {
        if (const Symbol * symbol{scope->symbol()}) {
          if (const auto *details{symbol->detailsIf<DerivedTypeDetails>()}) {
            for (const Symbol *param : details->paramDecls()) {
              if (const auto *details{param->detailsIf<TypeParamDetails>()}) {
                if (details->attr() == common::TypeParamAttr::Len) {
                  return true;
                }
              }
            }
          }
        }
      }
    } else if (type->category() == DeclTypeSpec::Category::TypeStar ||
        type->category() == DeclTypeSpec::Category::ClassStar) {
      return true;
    }
  }
  if (details.IsAssumedShape() || details.IsDeferredShape() ||
      details.IsAssumedRank()) {
    return true;
  }
  // TODO: Explicit shape component array dependent on length parameter
  // TODO: Automatic (adjustable) arrays - are they descriptors?
  return false;
}

static bool IsDescriptor(const ProcEntityDetails &details) {
  // A procedure pointer or dummy procedure must be & is a descriptor if
  // and only if it requires a static link.
  // TODO: refine this placeholder
  return details.HasExplicitInterface();
}

bool IsDescriptor(const Symbol &symbol) {
  if (const auto *objectDetails{symbol.detailsIf<ObjectEntityDetails>()}) {
    return IsAllocatableOrPointer(symbol) || IsDescriptor(*objectDetails);
  } else if (const auto *procDetails{symbol.detailsIf<ProcEntityDetails>()}) {
    if (symbol.attrs().test(Attr::POINTER) ||
        symbol.attrs().test(Attr::EXTERNAL)) {
      return IsDescriptor(*procDetails);
    }
  }
  return false;
}
}

namespace Fortran::evaluate {

template<typename A> bool PointeeComparison(const A *x, const A *y) {
  return x == y || (x != nullptr && y != nullptr && *x == *y);
}

bool DynamicType::operator==(const DynamicType &that) const {
  return category == that.category && kind == that.kind &&
      PointeeComparison(charLength, that.charLength) &&
      PointeeComparison(derived, that.derived);
}

bool DynamicType::IsAssumedLengthCharacter() const {
  return category == TypeCategory::Character && charLength != nullptr &&
      charLength->isAssumed();
}

static const semantics::DerivedTypeSpec *GetParentTypeSpec(
    const semantics::DerivedTypeSpec &spec) {
  const semantics::Symbol &typeSymbol{spec.typeSymbol()};
  if (const semantics::Scope * scope{typeSymbol.scope()}) {
    const auto &dtDetails{typeSymbol.get<semantics::DerivedTypeDetails>()};
    if (auto extends{dtDetails.GetParentComponentName()}) {
      if (auto iter{scope->find(*extends)}; iter != scope->cend()) {
        if (const Symbol & symbol{*iter->second};
            symbol.test(Symbol::Flag::ParentComp)) {
          return &symbol.get<semantics::ObjectEntityDetails>()
                      .type()
                      ->derivedTypeSpec();
        }
      }
    }
  }
  return nullptr;
}

static const bool IsAncestorTypeOf(const semantics::DerivedTypeSpec *ancestor,
    const semantics::DerivedTypeSpec *spec) {
  if (ancestor == nullptr) {
    return false;
  } else if (spec == nullptr) {
    return false;
  } else if (spec == ancestor) {
    return true;
  } else {
    return IsAncestorTypeOf(ancestor, GetParentTypeSpec(*spec));
  }
}

bool DynamicType::IsTypeCompatibleWith(const DynamicType &that) const {
  return *this == that || IsUnlimitedPolymorphic() ||
      (isPolymorphic && IsAncestorTypeOf(derived, that.derived));
}

std::optional<DynamicType> DynamicType::From(
    const semantics::DeclTypeSpec &type) {
  if (const auto *intrinsic{type.AsIntrinsic()}) {
    if (auto kind{ToInt64(intrinsic->kind())}) {
      TypeCategory category{intrinsic->category()};
      if (IsValidKindOfIntrinsicType(category, *kind)) {
        if (category == TypeCategory::Character) {
          const auto &charType{type.characterTypeSpec()};
          return DynamicType{static_cast<int>(*kind), charType.length()};
        } else {
          return DynamicType{category, static_cast<int>(*kind)};
        }
      }
    }
  } else if (const auto *derived{type.AsDerived()}) {
    return DynamicType{
        *derived, type.category() == semantics::DeclTypeSpec::ClassDerived};
  } else if (type.category() == semantics::DeclTypeSpec::ClassStar) {
    return DynamicType::UnlimitedPolymorphic();
  } else {
    // Assumed-type dummy arguments (TYPE(*)) do not have dynamic types.
  }
  return std::nullopt;
}

std::optional<DynamicType> DynamicType::From(const semantics::Symbol &symbol) {
  return From(symbol.GetType());  // Symbol -> DeclTypeSpec -> DynamicType
}

std::string DynamicType::AsFortran() const {
  if (derived != nullptr) {
    CHECK(category == TypeCategory::Derived);
    if (isPolymorphic) {
      return "CLASS("s + derived->typeSymbol().name().ToString() + ')';
    } else {
      return "TYPE("s + derived->typeSymbol().name().ToString() + ')';
    }
  } else if (charLength != nullptr) {
    std::string result{"CHARACTER(KIND="s + std::to_string(kind) + ",LEN="};
    if (charLength->isAssumed()) {
      result += '*';
    } else if (charLength->isDeferred()) {
      result += ':';
    } else if (const auto &length{charLength->GetExplicit()}) {
      std::stringstream ss;
      length->AsFortran(ss);
      result += ss.str();
    }
    return result + ')';
  } else if (isPolymorphic) {
    return "CLASS(*)";
  } else if (kind == 0) {
    return "(typeless intrinsic function argument)";
  } else {
    return EnumToString(category) + '(' + std::to_string(kind) + ')';
  }
}

std::string DynamicType::AsFortran(std::string &&charLenExpr) const {
  if (!charLenExpr.empty() && category == TypeCategory::Character) {
    return "CHARACTER(KIND=" + std::to_string(kind) +
        ",LEN=" + std::move(charLenExpr) + ')';
  } else {
    return AsFortran();
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

bool SomeKind<TypeCategory::Derived>::operator==(
    const SomeKind<TypeCategory::Derived> &that) const {
  return PointeeComparison(spec_, that.spec_);
}

std::string SomeDerived::AsFortran() const {
  std::stringstream out;
  DerivedTypeSpecAsFortran(out << "TYPE(", spec()) << ')';
  return out.str();
}
}
