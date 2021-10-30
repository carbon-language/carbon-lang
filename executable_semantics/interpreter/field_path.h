// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_FIELD_PATH_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_FIELD_PATH_H_

#include <string>
#include <vector>

#include "common/ostream.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

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

  // Constructs a FieldPath consisting of a single step.
  explicit FieldPath(std::string name) : components_({std::move(name)}) {}

  FieldPath(const FieldPath&) = default;
  FieldPath(FieldPath&&) = default;
  auto operator=(const FieldPath&) -> FieldPath& = default;
  auto operator=(FieldPath&&) -> FieldPath& = default;

  // Returns whether *this is empty.
  auto IsEmpty() const -> bool { return components_.empty(); }

  // Appends `name` to the end of *this.
  auto Append(std::string name) -> void {
    components_.push_back(std::move(name));
  }

  void Print(llvm::raw_ostream& out) const {
    for (const std::string& component : components_) {
      out << "." << component;
    }
  }

  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

 private:
  // The representation of FieldPath describes how to locate a Value within
  // another Value, so its implementation details are tied to the implementation
  // details of Value.
  friend class Value;
  std::vector<std::string> components_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_FIELD_PATH_H_
