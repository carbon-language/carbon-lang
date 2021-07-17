// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/function_definition.h"

namespace Carbon {

void FunctionDefinition::PrintDepth(llvm::raw_ostream& out, int depth) const {
  out << "fn " << name << " ";
  param_pattern->Print(out);
  out << " -> ";
  return_type->Print(out);
  if (body) {
    out << " {\n";
    body->Print(out, depth);
    out << "\n}\n";
  } else {
    out << ";\n";
  }
}

}  // namespace Carbon
