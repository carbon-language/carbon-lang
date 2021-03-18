// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/function_definition.h"

#include <iostream>

namespace Carbon {

auto MakeFunDef(int line_num, std::string name, Expression ret_type,
                Expression param_pattern, Statement* body)
    -> struct FunctionDefinition* {
  return new FunctionDefinition(line_num, std::move(name), param_pattern,
                                ret_type, body);
}

void PrintFunDefDepth(const FunctionDefinition* f, int depth) {
  std::cout << "fn " << f->name << " ";
  f->param_pattern.Print();
  std::cout << " -> ";
  f->return_type.Print();
  if (f->body) {
    std::cout << " {" << std::endl;
    PrintStatement(f->body, depth);
    std::cout << std::endl << "}" << std::endl;
  } else {
    std::cout << ";" << std::endl;
  }
}

void PrintFunDef(const FunctionDefinition* f) { PrintFunDefDepth(f, -1); }

}  // namespace Carbon
