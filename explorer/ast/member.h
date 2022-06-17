// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_MEMBER_H_
#define CARBON_EXPLORER_AST_MEMBER_H_

#include <optional>
#include <string>

#include "explorer/common/nonnull.h"
#include "llvm/ADT/PointerUnion.h"

namespace Carbon {

class Declaration;
class Value;

// A NamedValue represents a value with a name, such as a single struct field.
struct NamedValue {
  // The field name.
  std::string name;

  // The field's value.
  Nonnull<const Value*> value;
};

// A member of a type.
//
// This is either a declared member of a class, interface, or similar, or a
// member of a struct with no declaration.
class Member {
 public:
  explicit Member(Nonnull<const Declaration*> declaration);
  explicit Member(Nonnull<const NamedValue*> struct_member);

  // The name of the member.
  auto name() const -> std::string_view;
  // The declared type of the member, which might include type variables.
  auto type() const -> const Value&;
  // A declaration of the member, if any exists.
  auto declaration() const -> std::optional<Nonnull<const Declaration*>>;

 private:
  llvm::PointerUnion<Nonnull<const Declaration*>, Nonnull<const NamedValue*>>
      member_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_MEMBER_H_
