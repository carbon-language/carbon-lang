// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/field_path.h"

namespace Carbon {

namespace {
struct PrintVisitor {
  PrintVisitor(llvm::raw_ostream& out) : out(out) {}

  void operator()(const std::string& field_name) { out << "." << field_name; }

  void operator()(int index) { out << "[" << index << "]"; }

  llvm::raw_ostream& out;
};
}  // namespace

void FieldPath::Print(llvm::raw_ostream& out) const {
  for (const auto& component : components) {
    std::visit(PrintVisitor(out), component);
  }
}

}  // namespace Carbon
