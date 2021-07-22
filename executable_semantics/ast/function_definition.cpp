// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/function_definition.h"

namespace Carbon {

void FunctionDefinition::PrintDepth(int depth, llvm::raw_ostream& out) const {
  out << "fn " << name << " " << *param_pattern << " -> " << *return_type;
  if (body) {
    out << " {\n";
    body->PrintDepth(depth, out);
    out << "\n}\n";
  } else {
    out << ";\n";
  }
}

}  // namespace Carbon
