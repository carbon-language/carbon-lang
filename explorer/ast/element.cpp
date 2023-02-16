// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/element.h"

#include "common/check.h"
#include "explorer/ast/declaration.h"

namespace Carbon {
NamedElement::NamedElement(Nonnull<const Declaration*> declaration)
    : Element(ElementKind::NamedElement), element_(declaration) {}

NamedElement::NamedElement(Nonnull<const NamedValue*> struct_member)
    : Element(ElementKind::NamedElement), element_(struct_member) {}

auto NamedElement::IsNamed(std::string_view name) const -> bool {
  return this->name() == name;
}

auto NamedElement::name() const -> std::string_view {
  if (const auto* decl = element_.dyn_cast<const Declaration*>()) {
    return GetName(*decl).value();
  } else {
    const auto* named_value = element_.dyn_cast<const NamedValue*>();
    return named_value->name;
  }
}

auto NamedElement::type() const -> const Value& {
  if (const auto* decl = element_.dyn_cast<const Declaration*>()) {
    return decl->static_type();
  } else {
    const auto* named_value = element_.dyn_cast<const NamedValue*>();
    return *named_value->value;
  }
}

auto NamedElement::declaration() const
    -> std::optional<Nonnull<const Declaration*>> {
  if (const auto* decl = element_.dyn_cast<const Declaration*>()) {
    return decl;
  }
  return std::nullopt;
}

auto NamedElement::struct_member() const
    -> std::optional<Nonnull<const NamedValue*>> {
  if (const auto* member = element_.dyn_cast<const NamedValue*>()) {
    return member;
  }
  return std::nullopt;
}

void NamedElement::Print(llvm::raw_ostream& out) const { out << name(); }

// Prints the Element
void PositionalElement::Print(llvm::raw_ostream& out) const {
  out << "element #" << index_;
}

// Return whether the element's name matches `name`.
auto PositionalElement::IsNamed(std::string_view /*name*/) const -> bool {
  return false;
}

void BaseElement::Print(llvm::raw_ostream& out) const { out << "base class"; }

// Return whether the element's name matches `name`.
auto BaseElement::IsNamed(std::string_view /*name*/) const -> bool {
  return false;
}

}  // namespace Carbon
