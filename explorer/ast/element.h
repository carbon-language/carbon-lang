// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_ELEMENT_H_
#define CARBON_EXPLORER_AST_ELEMENT_H_

#include <optional>
#include <string>
#include <string_view>

#include "common/ostream.h"
#include "explorer/ast/ast_rtti.h"
#include "explorer/common/nonnull.h"
#include "llvm/ADT/PointerUnion.h"

namespace Carbon {

class Declaration;
class Value;

// A NamedValue represents a value with a name, such as a single struct field.
struct NamedValue {
  NamedValue(std::string name, Nonnull<const Value*> value)
      : name(std::move(name)), value(value) {}

  template <typename F>
  auto Decompose(F f) const {
    return f(name, value);
  }

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

  // Call `f` on this value, cast to its most-derived type. `R` specifies the
  // expected return type of `f`.
  template <typename R, typename F>
  auto Visit(F f) const -> R;

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

// A named element of a type.
//
// This is either a declared member of a class, interface, or similar, or a
// member of a struct with no declaration.
class NamedElement : public Element {
 public:
  explicit NamedElement(Nonnull<const Declaration*> declaration);
  explicit NamedElement(Nonnull<const NamedValue*> struct_member);

  template <typename F>
  auto Decompose(F f) const {
    if (auto decl = declaration()) {
      return f(*decl);
    } else {
      return f(*struct_member());
    }
  }

  // Prints the element's name
  void Print(llvm::raw_ostream& out) const override;

  auto IsNamed(std::string_view name) const -> bool override;

  static auto classof(const Element* member) -> bool {
    return InheritsFromNamedElement(member->kind());
  }

  auto type() const -> const Value& override;
  // The name of the member.
  auto name() const -> std::string_view;
  // A declaration of the member, if any exists.
  auto declaration() const -> std::optional<Nonnull<const Declaration*>>;
  // A name and type pair, if this is a struct member.
  auto struct_member() const -> std::optional<Nonnull<const NamedValue*>>;

 private:
  const llvm::PointerUnion<Nonnull<const Declaration*>,
                           Nonnull<const NamedValue*>>
      element_;
};

// A positional element of a type.
//
// This is a positional tuple element, or other index-based value.
class PositionalElement : public Element {
 public:
  explicit PositionalElement(int index, Nonnull<const Value*> type)
      : Element(ElementKind::PositionalElement), index_(index), type_(type) {}

  template <typename F>
  auto Decompose(F f) const {
    return f(index_, type_);
  }

  // Prints the element
  void Print(llvm::raw_ostream& out) const override;

  // Return whether the member's name matches `name`.
  auto IsNamed(std::string_view name) const -> bool override;

  static auto classof(const Element* member) -> bool {
    return InheritsFromPositionalElement(member->kind());
  }

  auto index() const -> int { return index_; }
  auto type() const -> const Value& override { return *type_; }

 private:
  const int index_;
  const Nonnull<const Value*> type_;
};

// A base class object.
//
// This is the base class object of a class value.
class BaseElement : public Element {
 public:
  explicit BaseElement(Nonnull<const Value*> type)
      : Element(ElementKind::BaseElement), type_(type) {}

  template <typename F>
  auto Decompose(F f) const {
    return f(type_);
  }

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

template <typename R, typename F>
auto Element::Visit(F f) const -> R {
  switch (kind()) {
    case ElementKind::NamedElement:
      return f(static_cast<const NamedElement*>(this));
    case ElementKind::PositionalElement:
      return f(static_cast<const PositionalElement*>(this));
    case ElementKind::BaseElement:
      return f(static_cast<const BaseElement*>(this));
  }
}

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_ELEMENT_H_
