// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/syntax_helpers.h"

#include <iostream>

#include "executable_semantics/interpreter/interpreter.h"
#include "executable_semantics/interpreter/typecheck.h"
#include "executable_semantics/tracing_flag.h"

namespace Carbon {

void ExecProgram(std::list<Declaration>* fs) {
  if (tracing_output) {
    std::cout << "********** source program **********" << std::endl;
    for (const auto& decl : *fs) {
      decl.Print();
    }
    std::cout << "********** type checking **********" << std::endl;
  }
  state = new State();  // Compile-time state.
  TypeCheckContext p = TopLevel(fs);
  TypeEnv top = p.types;
  Env ct_top = p.values;
  std::list<Declaration> new_decls;
  for (const auto& decl : *fs) {
    new_decls.push_back(MakeTypeChecked(decl, top, ct_top));
  }
  if (tracing_output) {
    std::cout << std::endl;
    std::cout << "********** type checking complete **********" << std::endl;
    for (const auto& decl : new_decls) {
      decl.Print();
    }
    std::cout << "********** starting execution **********" << std::endl;
  }
  int result = InterpProgram(&new_decls);
  std::cout << "result: " << result << std::endl;
}

}  // namespace Carbon
