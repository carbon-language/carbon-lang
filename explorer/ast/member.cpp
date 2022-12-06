// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/member.h"

#include "common/check.h"
#include "explorer/ast/declaration.h"

namespace Carbon {

Member::Member(Nonnull<const Declaration*> declaration)
    : member_(declaration) {}

Member::Member(Nonnull<const NamedValue*> struct_member)
    : member_(struct_member) {}

Member::Member(Nonnull<const IndexedValue*> tuple_member)
    : member_(tuple_member) {}

auto Member::IsNamed(std::string_view other_name) const -> bool {
  return HasName() && name() == other_name;
}

auto Member::name() const -> std::string_view {
  CARBON_CHECK(HasName()) << "Unnamed member does not have a name()";
  if (const auto* decl = member_.dyn_cast<const Declaration*>()) {
    return GetName(*decl).value();
  } else if (const auto* named_valued = member_.dyn_cast<const NamedValue*>()) {
    return named_valued->name;
  } else {
    CARBON_FATAL() << "Unreachable";
  }
}

auto Member::HasPosition() const -> bool {
  return member_.dyn_cast<const IndexedValue*>() != nullptr;
}

auto Member::HasName() const -> bool {
  // Both are currently mutually exclusive
  return !HasPosition();
}

auto Member::index() const -> int {
  CARBON_CHECK(HasPosition())
      << "Non-positional member does not have an index()";
  return member_.dyn_cast<const IndexedValue*>()->index;
}

auto Member::type() const -> const Value& {
  if (const auto* decl = member_.dyn_cast<const Declaration*>()) {
    return decl->static_type();
  } else if (const auto* named_valued = member_.dyn_cast<const NamedValue*>()) {
    return *named_valued->value;
  } else {
    return *member_.get<const IndexedValue*>()->value;
  }
}

auto Member::declaration() const -> std::optional<Nonnull<const Declaration*>> {
  if (const auto* decl = member_.dyn_cast<const Declaration*>()) {
    return decl;
  }
  return std::nullopt;
}

void Member::Print(llvm::raw_ostream& out) const {
  if (HasName()) {
    out << name();
  } else if (const auto* value = member_.dyn_cast<const IndexedValue*>()) {
    out << "element #" << member_.get<const IndexedValue*>()->index;
  } else {
    CARBON_FATAL() << "Unhandled member type";
  }
}

}  // namespace Carbon
