// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPLORER_INTERPRETER_FIELD_PATH_H_
#define EXPLORER_INTERPRETER_FIELD_PATH_H_

#include <string>
#include <vector>

#include "common/ostream.h"
#include "explorer/ast/static_scope.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class Witness;

// Given some initial Value, a FieldPath identifies a sub-Value within it,
// in much the same way that a file path identifies a file within some
// directory. FieldPaths are relative rather than absolute: the initial
// Value is specified by the context in which the FieldPath is used, not
// by the FieldPath itself.
//
// A FieldPath consists of a series of steps, which specify how to
// incrementally navigate from a Value to one of its fields. Currently
// there is only one kind of step, a string specifying a child field by name,
// but that may change as Carbon develops. Note that an empty FieldPath
// refers to the initial Value itself.
class FieldPath {
 public:
  // Constructs an empty FieldPath.
  FieldPath() = default;

  // A single component of the FieldPath, which is typically the name
  // of a field. However, inside a generic, when there is a field
  // access on something of a generic type, e.g., `T`, then we also
  // need `witness`, a pointer to the witness table containing that field.
  class Component {
   public:
    explicit Component(std::string name) : name_(std::move(name)) {}
    Component(std::string name, std::optional<Nonnull<const Witness*>> witness)
        : name_(std::move(name)), witness_(witness) {}

    auto name() const -> const std::string& { return name_; }

    auto witness() const -> std::optional<Nonnull<const Witness*>> {
      return witness_;
    }

    void Print(llvm::raw_ostream& out) const { out << name_; }

   private:
    std::string name_;
    std::optional<Nonnull<const Witness*>> witness_;
  };

  // Constructs a FieldPath consisting of a single step.
  explicit FieldPath(std::string name)
      : components_({Component(std::move(name))}) {}
  explicit FieldPath(const Component& f) : components_({f}) {}

  FieldPath(const FieldPath&) = default;
  FieldPath(FieldPath&&) = default;
  auto operator=(const FieldPath&) -> FieldPath& = default;
  auto operator=(FieldPath&&) -> FieldPath& = default;

  auto components() const -> llvm::ArrayRef<Component> { return components_; }

  // Returns whether *this is empty.
  auto IsEmpty() const -> bool { return components_.empty(); }

  // Appends `name` to the end of *this.
  auto Append(std::string name) -> void {
    components_.push_back(Component(std::move(name)));
  }

  void Print(llvm::raw_ostream& out) const {
    for (const Component& component : components_) {
      out << "." << component;
    }
  }

  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

 private:
  // The representation of FieldPath describes how to locate a Value within
  // another Value, so its implementation details are tied to the implementation
  // details of Value.
  friend class Value;
  std::vector<Component> components_;
};

}  // namespace Carbon

#endif  // EXPLORER_INTERPRETER_FIELD_PATH_H_
