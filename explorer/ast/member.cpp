// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/member.h"

#include "explorer/ast/declaration.h"

namespace Carbon {

Member::Member(Nonnull<const Declaration*> declaration)
    : member_(declaration) {}

Member::Member(Nonnull<const NamedValue*> struct_member)
    : member_(struct_member) {}

Member::Member(Nonnull<const IndexedValue*> tuple_member)
    : member_(tuple_member) {}

auto Member::name() const -> std::string_view {
  if (const auto* decl = member_.dyn_cast<const Declaration*>()) {
    return GetName(*decl).value();
  } else if (const auto* named_valued = member_.dyn_cast<const NamedValue*>()) {
    return named_valued->name;
  } else {
    return "";
  }
}

auto Member::index() const -> std::optional<size_t> {
  if (const auto* value = member_.dyn_cast<const IndexedValue*>()) {
    return value->index;
  } else {
    return std::nullopt;
  }
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

}  // namespace Carbon
