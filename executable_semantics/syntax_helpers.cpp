// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax_helpers.h"

#include <iostream>

#include "executable_semantics/interpreter/interpreter.h"
#include "executable_semantics/interpreter/typecheck.h"

namespace Carbon {

const char* input_filename = nullptr;

void PrintSyntaxError(char* error, int line_num) {
  std::cerr << input_filename << ":" << line_num << ": " << error << std::endl;
  exit(-1);
}

void ExecProgram(std::list<Declaration>* fs) {
  std::cout << "********** source program **********" << std::endl;
  for (const auto& decl : *fs) {
    decl.Print();
  }
  std::cout << "********** type checking **********" << std::endl;
  state = new State();  // Compile-time state.
  std::pair<TypeEnv*, Env*> p = TopLevel(fs);
  TypeEnv* top = p.first;
  Env* ct_top = p.second;
  std::list<Declaration> new_decls;
  for (const auto& decl : *fs) {
    new_decls.push_back(decl.TypeChecked(top, ct_top));
  }
  std::cout << std::endl;
  std::cout << "********** type checking complete **********" << std::endl;
  for (const auto& decl : new_decls) {
    decl.Print();
  }
  std::cout << "********** starting execution **********" << std::endl;
  int result = InterpProgram(&new_decls);
  std::cout << "result: " << result << std::endl;
}

}  // namespace Carbon
