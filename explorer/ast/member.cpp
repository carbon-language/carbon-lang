// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/member.h"

#include "common/check.h"
#include "explorer/ast/declaration.h"

namespace Carbon {
NominalMember::NominalMember(Nonnull<const Declaration*> declaration)
    : Member(MemberKind::NominalMember, GetName(*declaration).value()),
      member_(declaration) {
  CARBON_CHECK(name_) << "Missing name for NominalMember";
}

NominalMember::NominalMember(Nonnull<const NamedValue*> struct_member)
    : Member(MemberKind::NominalMember, struct_member->name),
      member_(struct_member) {}

NominalMember::NominalMember(const NominalMember& other)
    : Member(other.kind(), other.name_.value()), member_(other.member_) {}

NominalMember::NominalMember(NominalMember&& other) noexcept
    : Member(other.kind(), other.name_.value()), member_(other.member_) {}

auto NominalMember::name() const -> std::string_view { return name_.value(); }

auto NominalMember::type() const -> const Value& {
  if (const auto* decl = member_.dyn_cast<const Declaration*>()) {
    return decl->static_type();
  } else {
    const auto* named_valued = member_.dyn_cast<const NamedValue*>();
    return *named_valued->value;
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

void BaseClass::Print(llvm::raw_ostream& out) const { out << "base class"; }

}  // namespace Carbon
