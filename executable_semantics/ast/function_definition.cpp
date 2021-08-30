// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/function_definition.h"

namespace Carbon {

void FunctionDefinition::PrintDepth(int depth, llvm::raw_ostream& out) const {
  out << "fn " << name << " ";
  if (deduced_parameters.size() > 0) {
    out << "[";
    unsigned int i = 0;
    for (const auto& deduced : deduced_parameters) {
      if (i != 0) {
        out << ", ";
      }
      out << deduced.name << ":! ";
      deduced.type->Print(out);
      ++i;
    }
    out << "]";
  }
  out << *param_pattern;
  if (!is_omitted_return_type) {
    out << " -> " << *return_type;
  }
  if (body) {
    out << " {\n";
    (*body)->PrintDepth(depth, out);
    out << "\n}\n";
  } else {
    out << ";\n";
  }
}

}  // namespace Carbon
