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
#include "../common/template.h"
#include "../parser/characters.h"
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

bool IsDescriptor(const Symbol &symbol0) {
  const Symbol &symbol{evaluate::ResolveAssociations(symbol0)};
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

template<typename A> inline bool PointeeComparison(const A *x, const A *y) {
  return x == y || (x != nullptr && y != nullptr && *x == *y);
}

bool DynamicType::operator==(const DynamicType &that) const {
  return category_ == that.category_ && kind_ == that.kind_ &&
      PointeeComparison(charLength_, that.charLength_) &&
      PointeeComparison(derived_, that.derived_);
}

bool DynamicType::IsAssumedLengthCharacter() const {
  return category_ == TypeCategory::Character && charLength_ != nullptr &&
      charLength_->isAssumed();
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
  } else if (spec->typeSymbol() == ancestor->typeSymbol()) {
    return true;
  } else {
    return IsAncestorTypeOf(ancestor, GetParentTypeSpec(*spec));
  }
}

bool DynamicType::IsTypeCompatibleWith(const DynamicType &that) const {
  return *this == that || IsUnlimitedPolymorphic() ||
      (IsPolymorphic() && derived_ != nullptr &&
          IsAncestorTypeOf(derived_, that.derived_));
}

// Do the kind type parameters of type1 have the same values as the
// corresponding kind type parameters of the type2?
static bool IsKindCompatible(const semantics::DerivedTypeSpec &type1,
    const semantics::DerivedTypeSpec &type2) {
  for (const auto &[name, param1] : type1.parameters()) {
    if (param1.isKind()) {
      const semantics::ParamValue *param2{type2.FindParameter(name)};
      if (!PointeeComparison(&param1, param2)) {
        return false;
      }
    }
  }
  return true;
}

bool DynamicType::IsTkCompatibleWith(const DynamicType &that) const {
  if (category_ != TypeCategory::Derived) {
    return category_ == that.category_ && kind_ == that.kind_;
  } else if (IsUnlimitedPolymorphic()) {
    return true;
  } else if (that.IsUnlimitedPolymorphic()) {
    return false;
  } else if (!derived_ || !that.derived_ ||
      !IsKindCompatible(*derived_, *that.derived_)) {
    return false;  // kind params don't match
  } else if (!IsPolymorphic()) {
    return derived_->typeSymbol() == that.derived_->typeSymbol();
  } else {
    return IsAncestorTypeOf(derived_, that.derived_);
  }
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
  } else if (type.category() == semantics::DeclTypeSpec::TypeStar) {
    return DynamicType::AssumedType();
  } else {
    common::die("DynamicType::From(DeclTypeSpec): failed");
  }
  return std::nullopt;
}

std::optional<DynamicType> DynamicType::From(const semantics::Symbol &symbol) {
  return From(symbol.GetType());  // Symbol -> DeclTypeSpec -> DynamicType
}

DynamicType DynamicType::ResultTypeForMultiply(const DynamicType &that) const {
  switch (category_) {
  case TypeCategory::Integer:
    switch (that.category_) {
    case TypeCategory::Integer:
      return DynamicType{TypeCategory::Integer, std::max(kind_, that.kind_)};
    case TypeCategory::Real:
    case TypeCategory::Complex: return that;
    default: CRASH_NO_CASE;
    }
    break;
  case TypeCategory::Real:
    switch (that.category_) {
    case TypeCategory::Integer: return *this;
    case TypeCategory::Real:
      return DynamicType{TypeCategory::Real, std::max(kind_, that.kind_)};
    case TypeCategory::Complex:
      return DynamicType{TypeCategory::Complex, std::max(kind_, that.kind_)};
    default: CRASH_NO_CASE;
    }
    break;
  case TypeCategory::Complex:
    switch (that.category_) {
    case TypeCategory::Integer: return *this;
    case TypeCategory::Real:
    case TypeCategory::Complex:
      return DynamicType{TypeCategory::Complex, std::max(kind_, that.kind_)};
    default: CRASH_NO_CASE;
    }
    break;
  case TypeCategory::Logical:
    switch (that.category_) {
    case TypeCategory::Logical:
      return DynamicType{TypeCategory::Logical, std::max(kind_, that.kind_)};
    default: CRASH_NO_CASE;
    }
    break;
  default: CRASH_NO_CASE;
  }
  return *this;
}

bool SomeKind<TypeCategory::Derived>::operator==(
    const SomeKind<TypeCategory::Derived> &that) const {
  return PointeeComparison(derivedTypeSpec_, that.derivedTypeSpec_);
}

int SelectedCharKind(const std::string &s) {  // 16.9.168
  auto lower{parser::ToLowerCaseLetters(s)};
  auto n{lower.size()};
  while (n > 0 && lower[0] == ' ') {
    lower.erase(0, 1);
    --n;
  }
  while (n > 0 && lower[n - 1] == ' ') {
    lower.erase(--n, 1);
  }
  if (lower == "ascii") {
    return 1;
  } else if (lower == "ucs-2") {
    return 2;
  } else if (lower == "iso_10646" || lower == "ucs-4") {
    return 4;
  } else {
    return -1;
  }
}

class SelectedIntKindVisitor {
public:
  explicit SelectedIntKindVisitor(std::int64_t p) : precision_{p} {}
  using Result = std::optional<int>;
  using Types = IntegerTypes;
  template<typename T> Result Test() const {
    if (Scalar<T>::RANGE >= precision_) {
      return T::kind;
    } else {
      return std::nullopt;
    }
  }

private:
  std::int64_t precision_;
};

int SelectedIntKind(std::int64_t precision) {
  if (auto kind{common::SearchTypes(SelectedIntKindVisitor{precision})}) {
    return *kind;
  } else {
    return -1;
  }
}

class SelectedRealKindVisitor {
public:
  explicit SelectedRealKindVisitor(std::int64_t p, std::int64_t r)
    : precision_{p}, range_{r} {}
  using Result = std::optional<int>;
  using Types = RealTypes;
  template<typename T> Result Test() const {
    if (Scalar<T>::PRECISION >= precision_ && Scalar<T>::RANGE >= range_) {
      return {T::kind};
    } else {
      return std::nullopt;
    }
  }

private:
  std::int64_t precision_, range_;
};

int SelectedRealKind(
    std::int64_t precision, std::int64_t range, std::int64_t radix) {
  if (radix != 2) {
    return -5;
  }
  if (auto kind{
          common::SearchTypes(SelectedRealKindVisitor{precision, range})}) {
    return *kind;
  }
  // No kind has both sufficient precision and sufficient range.
  // The negative return value encodes whether any kinds exist that
  // could satisfy either constraint independently.
  bool pOK{common::SearchTypes(SelectedRealKindVisitor{precision, 0})};
  bool rOK{common::SearchTypes(SelectedRealKindVisitor{0, range})};
  if (pOK) {
    if (rOK) {
      return -4;
    } else {
      return -2;
    }
  } else {
    if (rOK) {
      return -1;
    } else {
      return -3;
    }
  }
}
}
