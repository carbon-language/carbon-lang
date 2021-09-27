// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/function_definition.h"

namespace Carbon {

void FunctionDefinition::PrintDepth(int depth, llvm::raw_ostream& out) const {
  out << "fn " << name_ << " ";
  if (!deduced_parameters_.empty()) {
    out << "[";
    unsigned int i = 0;
    for (const auto& deduced : deduced_parameters_) {
      if (i != 0) {
        out << ", ";
      }
      out << deduced.name << ":! ";
      deduced.type->Print(out);
      ++i;
    }
    out << "]";
  }
  out << *param_pattern_;
  if (!is_omitted_return_type_) {
    out << " -> " << *return_type_;
  }
  if (body_) {
    out << " {\n";
    (*body_)->PrintDepth(depth, out);
    out << "\n}\n";
  } else {
    out << ";\n";
  }
}

}  // namespace Carbon
