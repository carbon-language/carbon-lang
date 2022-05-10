// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/exec_program.h"

#include <variant>

#include "common/check.h"
#include "common/ostream.h"
#include "explorer/common/arena.h"
#include "explorer/interpreter/interpreter.h"
#include "explorer/interpreter/resolve_control_flow.h"
#include "explorer/interpreter/resolve_names.h"
#include "explorer/interpreter/type_checker.h"
#include "llvm/Support/Error.h"

namespace Carbon {

auto ExecProgram(Nonnull<Arena*> arena, AST ast, bool trace) -> ErrorOr<int> {
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
  RETURN_IF_ERROR(ResolveNames(ast));
  if (trace) {
    llvm::outs() << "********** resolving control flow **********\n";
  }
  RETURN_IF_ERROR(ResolveControlFlow(ast));
  if (trace) {
    llvm::outs() << "********** type checking **********\n";
  }
  RETURN_IF_ERROR(TypeChecker(arena, trace).TypeCheck(ast));
  if (trace) {
    llvm::outs() << "\n";
    llvm::outs() << "********** type checking complete **********\n";
    for (const auto decl : ast.declarations) {
      llvm::outs() << *decl;
    }
    llvm::outs() << "********** starting execution **********\n";
  }
  ASSIGN_OR_RETURN(const int result, InterpProgram(ast, arena, trace));
  llvm::outs() << "result: " << result << "\n";
  return result;
}

}  // namespace Carbon
