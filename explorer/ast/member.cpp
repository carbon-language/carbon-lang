// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/member.h"

#include <optional>
#include <string_view>

#include "common/check.h"
#include "explorer/ast/declaration.h"

namespace Carbon {

Member::Member(Nonnull<const Declaration*> declaration)
    : member_(declaration) {}

Member::Member(Nonnull<const NamedValue*> struct_member)
    : member_(struct_member) {}

Member::Member(Nonnull<const IndexedValue*> tuple_member)
    : member_(tuple_member) {}

auto Member::name() const -> std::optional<std::string_view> {
  if (const auto* decl = member_.dyn_cast<const Declaration*>()) {
    return GetName(*decl).value();
  } else if (const auto* named_valued = member_.dyn_cast<const NamedValue*>()) {
    return named_valued->name;
  } else {
    return std::nullopt;
  }
}

auto Member::IsNamed(std::string_view other_name) const -> bool {
  const auto member_name = name();
  return member_name && other_name == member_name;
}

auto Member::isPositional() const -> bool {
  return member_.dyn_cast<const IndexedValue*>() != nullptr;
}

auto Member::index() const -> size_t {
  const auto* value = member_.dyn_cast<const IndexedValue*>();
  CARBON_CHECK(value)
      << "Member::index() requires to be used with a positional member";
  return value->index;
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
  const auto member_name = name();
  if (member_name) {
    out << member_name.value();
  } else if (const auto* value = member_.dyn_cast<const IndexedValue*>()) {
    out << "element #" << member_.get<const IndexedValue*>()->index;
  } else {
    CARBON_FATAL() << "Unhandled member type";
  }
}

}  // namespace Carbon
