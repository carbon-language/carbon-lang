// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_ELEMENT_PATH_H_
#define CARBON_EXPLORER_INTERPRETER_ELEMENT_PATH_H_

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common/ostream.h"
#include "explorer/ast/element.h"
#include "explorer/ast/static_scope.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class InterfaceType;
class Witness;

// Given some initial Value, a ElementPath identifies a sub-Value within it,
// in much the same way that a file path identifies a file within some
// directory. FieldPaths are relative rather than absolute: the initial
// Value is specified by the context in which the ElementPath is used, not
// by the ElementPath itself.
//
// A ElementPath consists of a series of steps, which specify how to
// incrementally navigate from a Value to one of its fields. Currently
// there is only one kind of step, a string specifying a child field by name,
// but that may change as Carbon develops. Note that an empty ElementPath
// refers to the initial Value itself.
class ElementPath {
 public:
  // Constructs an empty ElementPath.
  ElementPath() = default;

  // A single component of the ElementPath, which is typically the name
  // of a field. However, inside a generic, when there is a field
  // access on something of a generic type, e.g., `T`, then we also
  // need `witness`, a pointer to the witness table containing that field.
  class Component {
   public:
    explicit Component(Nonnull<const Element*> element) : element_(element) {}
    Component(Nonnull<const Element*> element,
              std::optional<Nonnull<const InterfaceType*>> interface,
              std::optional<Nonnull<const Witness*>> witness)
        : element_(element), interface_(interface), witness_(witness) {}

    auto element() const -> Nonnull<const Element*> { return element_; }

    auto IsNamed(std::string_view name) const -> bool {
      return element_->IsNamed(name);
    }

    auto interface() const -> std::optional<Nonnull<const InterfaceType*>> {
      return interface_;
    }

    auto witness() const -> std::optional<Nonnull<const Witness*>> {
      return witness_;
    }

    void Print(llvm::raw_ostream& out) const { return element_->Print(out); }

   private:
    Nonnull<const Element*> element_;
    std::optional<Nonnull<const InterfaceType*>> interface_;
    std::optional<Nonnull<const Witness*>> witness_;
  };

  // Constructs a ElementPath consisting of a single step.
  explicit ElementPath(Nonnull<const Element*> element)
      : components_({Component(element)}) {}
  explicit ElementPath(const Component& f) : components_({f}) {}

  ElementPath(const ElementPath&) = default;
  ElementPath(ElementPath&&) = default;
  auto operator=(const ElementPath&) -> ElementPath& = default;
  auto operator=(ElementPath&&) -> ElementPath& = default;

  // Returns whether *this is empty.
  auto IsEmpty() const -> bool { return components_.empty(); }

  // Appends `element` to the end of *this.
  auto Append(Nonnull<const Element*> element) -> void {
    components_.push_back(Component(element));
  }

  void Print(llvm::raw_ostream& out) const {
    for (const Component& component : components_) {
      out << "." << component;
    }
  }

  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

 private:
  // The representation of ElementPath describes how to locate a Value within
  // another Value, so its implementation details are tied to the implementation
  // details of Value.
  friend class Value;
  std::vector<Component> components_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_ELEMENT_PATH_H_
