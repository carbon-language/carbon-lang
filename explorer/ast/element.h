// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_ELEMENT_H_
#define CARBON_EXPLORER_AST_ELEMENT_H_

#include <optional>
#include <string>
#include <string_view>

#include "explorer/ast/ast_rtti.h"
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

// A generic member of a type.
//
// This is can be a named, positional or other type of member.
class Element {
 protected:
  explicit Element(ElementKind kind) : kind_(kind) {}

 public:
  virtual ~Element() = default;

  // Prints the Member
  virtual void Print(llvm::raw_ostream& out) const = 0;

  // Return whether the member's name matches `name`.
  virtual auto IsNamed(std::string_view name) const -> bool = 0;

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> ElementKind { return kind_; }

  // The declared type of the member, which might include type variables.
  virtual auto type() const -> const Value& = 0;

 private:
  const ElementKind kind_;
};

// A named member of a type.
//
// This is either a declared member of a class, interface, or similar, or a
// member of a struct with no declaration.
class MemberElement : public Element {
 public:
  explicit MemberElement(Nonnull<const Declaration*> declaration);
  explicit MemberElement(Nonnull<const NamedValue*> struct_member);

  // Prints the Member
  void Print(llvm::raw_ostream& out) const override;

  auto IsNamed(std::string_view name) const -> bool override;

  static auto classof(const Element* member) -> bool {
    return InheritsFromMemberElement(member->kind());
  }

  auto type() const -> const Value& override;
  // The name of the member.
  auto name() const -> std::string_view;
  // A declaration of the member, if any exists.
  auto declaration() const -> std::optional<Nonnull<const Declaration*>>;

 private:
  const llvm::PointerUnion<Nonnull<const Declaration*>,
                           Nonnull<const NamedValue*>>
      element_;
};

// A positional member of a type.
//
// This is a member of a tuple, or other index-based value.
class TupleElement : public Element {
 public:
  explicit TupleElement(size_t index, Nonnull<const Value*> type)
      : Element(ElementKind::TupleElement), index_(index), type_(type) {}

  // Prints the Member
  void Print(llvm::raw_ostream& out) const override;

  // Return whether the member's name matches `name`.
  auto IsNamed(std::string_view name) const -> bool override;

  static auto classof(const Element* member) -> bool {
    return InheritsFromTupleElement(member->kind());
  }

  auto index() const -> size_t { return index_; }
  auto type() const -> const Value& override { return *type_; }

 private:
  const size_t index_;
  const Nonnull<const Value*> type_;
};

// A base class object.
//
// This is the base class object of a class value.
class BaseElement : public Element {
 public:
  explicit BaseElement(Nonnull<const Value*> type)
      : Element(ElementKind::BaseElement), type_(type) {}

  // Prints the Member
  void Print(llvm::raw_ostream& out) const override;

  // Return whether the member's name matches `name`.
  auto IsNamed(std::string_view name) const -> bool override;

  static auto classof(const Element* member) -> bool {
    return InheritsFromBaseElement(member->kind());
  }

  auto type() const -> const Value& override { return *type_; }

 private:
  const Nonnull<const Value*> type_;
};
}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_ELEMENT_H_
