// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/exec_program.h"

#include <variant>

#include "common/check.h"
#include "common/ostream.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/interpreter/interpreter.h"
#include "executable_semantics/interpreter/resolve_control_flow.h"
#include "executable_semantics/interpreter/resolve_names.h"
#include "executable_semantics/interpreter/type_checker.h"

namespace Carbon {

void ExecProgram(Nonnull<Arena*> arena, AST ast, bool trace) {
  if (trace) {
    llvm::outs() << "********** source program **********\n";
    for (const auto decl : ast.declarations) {
      llvm::outs() << *decl;
    }
  }
  SourceLocation source_loc("<Main()>", 0);
  ast.main_call = arena->New<CallExpression>(
      source_loc, arena->New<IdentifierExpression>(source_loc, "Main"),
      arena->New<TupleLiteral>(source_loc));
  // Although name resolution is currently done once, generic programming
  // (particularly templates) may require more passes.
  if (trace) {
    llvm::outs() << "********** resolving names **********\n";
  }
  ResolveNames(ast);
  if (trace) {
    llvm::outs() << "********** resolving control flow **********\n";
  }
  ResolveControlFlow(ast);
  if (trace) {
    llvm::outs() << "********** type checking **********\n";
  }
  TypeChecker(arena, trace).TypeCheck(ast);
  if (trace) {
    llvm::outs() << "\n";
    llvm::outs() << "********** type checking complete **********\n";
    for (const auto decl : ast.declarations) {
      llvm::outs() << *decl;
    }
    llvm::outs() << "********** starting execution **********\n";
  }
  int result = InterpProgram(ast, arena, trace);
  llvm::outs() << "result: " << result << "\n";
}

}  // namespace Carbon
