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

auto Member::name() const -> std::string_view {
  if (const auto* decl = member_.dyn_cast<const Declaration*>()) {
    return GetName(*decl).value();
  } else {
    return member_.get<const NamedValue*>()->name;
  }
}

auto Member::type() const -> const Value& {
  if (const auto* decl = member_.dyn_cast<const Declaration*>()) {
    return decl->static_type();
  } else {
    return *member_.get<const NamedValue*>()->value;
  }
}

auto Member::declaration() const -> std::optional<Nonnull<const Declaration*>> {
  if (const auto* decl = member_.dyn_cast<const Declaration*>()) {
    return decl;
  }
  return std::nullopt;
}

}  // namespace Carbon
