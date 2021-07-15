// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/function_definition.h"

#include <iostream>

namespace Carbon {

void FunctionDefinition::PrintDepth(int depth) const {
  std::cout << "fn " << name << " ";
  PrintExp(param_pattern);
  std::cout << " -> ";
  PrintExp(return_type);
  if (body) {
    std::cout << " {" << std::endl;
    PrintStatement(body, depth);
    std::cout << std::endl << "}" << std::endl;
  } else {
    std::cout << ";" << std::endl;
  }
}

}  // namespace Carbon
