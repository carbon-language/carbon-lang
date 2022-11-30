// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/member.h"

#include "common/check.h"
#include "explorer/ast/declaration.h"

namespace Carbon {
NominalMember::NominalMember(Nonnull<const Declaration*> declaration)
    : Member(MemberKind::NominalMember), member_(declaration) {}

NominalMember::NominalMember(Nonnull<const NamedValue*> struct_member)
    : Member(MemberKind::NominalMember), member_(struct_member) {}

auto NominalMember::IsNamed(std::string_view name) const -> bool {
  return this->name() == name;
}

auto NominalMember::name() const -> std::string_view {
  if (const auto* decl = member_.dyn_cast<const Declaration*>()) {
    return GetName(*decl).value();
  } else {
    const auto* named_value = member_.dyn_cast<const NamedValue*>();
    return named_value->name;
  }
}

auto NominalMember::type() const -> const Value& {
  if (const auto* decl = member_.dyn_cast<const Declaration*>()) {
    return decl->static_type();
  } else {
    const auto* named_value = member_.dyn_cast<const NamedValue*>();
    return *named_value->value;
  }
}

auto NominalMember::declaration() const
    -> std::optional<Nonnull<const Declaration*>> {
  if (const auto* decl = member_.dyn_cast<const Declaration*>()) {
    return decl;
  }
  return std::nullopt;
}

void NominalMember::Print(llvm::raw_ostream& out) const { out << name(); }

// Prints the Member
void PositionalMember::Print(llvm::raw_ostream& out) const {
  out << "element #" << index_;
}

// Return whether the member's name matches `name`.
auto PositionalMember::IsNamed(std::string_view /*name*/) const -> bool {
  return false;
}

void BaseClassObjectMember::Print(llvm::raw_ostream& out) const {
  out << "base class";
}

// Return whether the member's name matches `name`.
auto BaseClassObjectMember::IsNamed(std::string_view /*name*/) const -> bool {
  return false;
}

}  // namespace Carbon
