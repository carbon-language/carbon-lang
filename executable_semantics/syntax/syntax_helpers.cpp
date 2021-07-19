// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/syntax_helpers.h"

#include "executable_semantics/interpreter/interpreter.h"
#include "executable_semantics/interpreter/typecheck.h"
#include "executable_semantics/tracing_flag.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

void ExecProgram(std::list<Declaration>* fs) {
  if (tracing_output) {
    llvm::outs() << "********** source program **********\n";
    for (const auto& decl : *fs) {
      decl.Print(llvm::errs());
    }
    llvm::outs() << "********** type checking **********\n";
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
    llvm::outs() << "\n";
    llvm::outs() << "********** type checking complete **********\n";
    for (const auto& decl : new_decls) {
      decl.Print(llvm::outs());
    }
    llvm::outs() << "********** starting execution **********\n";
  }
  int result = InterpProgram(&new_decls);
  llvm::outs() << "result: " << result << "\n";
}

}  // namespace Carbon
