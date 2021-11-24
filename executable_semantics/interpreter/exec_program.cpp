// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/exec_program.h"

#include "common/check.h"
#include "common/ostream.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/interpreter/interpreter.h"
#include "executable_semantics/interpreter/resolve_control_flow.h"
#include "executable_semantics/interpreter/resolve_names.h"
#include "executable_semantics/interpreter/type_checker.h"

namespace Carbon {

// Adds builtins, currently only Print(). Note Print() is experimental, not
// standardized, but is made available for printing state in tests.
static void AddIntrinsics(Nonnull<Arena*> arena,
                          std::vector<Nonnull<Declaration*>>* declarations) {
  SourceLocation source_loc("<intrinsic>", 0);
  std::vector<Nonnull<Pattern*>> print_params = {arena->New<BindingPattern>(
      source_loc, "format_str",
      arena->New<ExpressionPattern>(
          arena->New<StringTypeLiteral>(source_loc)))};
  auto print_return = arena->New<Block>(
      source_loc, std::vector<Nonnull<Statement*>>({arena->New<Return>(
                      source_loc,
                      arena->New<IntrinsicExpression>(
                          IntrinsicExpression::Intrinsic::Print),
                      false)}));
  auto print = arena->New<FunctionDeclaration>(
      source_loc, "Print", std::vector<Nonnull<GenericBinding*>>(),
      arena->New<TuplePattern>(source_loc, print_params),
      ReturnTerm::Explicit(arena->New<TupleLiteral>(source_loc)), print_return);
  declarations->insert(declarations->begin(), print);
}

void ExecProgram(Nonnull<Arena*> arena, AST ast, bool trace) {
  AddIntrinsics(arena, &ast.declarations);
  if (trace) {
    llvm::outs() << "********** source program **********\n";
    for (const auto decl : ast.declarations) {
      llvm::outs() << *decl;
    }
    llvm::outs() << "********** type checking **********\n";
  }
  SourceLocation source_loc("<Main()>", 0);
  ast.main_call = arena->New<CallExpression>(
      source_loc, arena->New<IdentifierExpression>(source_loc, "Main"),
      arena->New<TupleLiteral>(source_loc));
  // Although name resolution is currently done once, generic programming
  // (particularly templates) may require more passes.
  ResolveNames(arena, ast);
  ResolveControlFlow(ast);
  TypeChecker(arena, trace).TypeCheck(ast);
  if (trace) {
    llvm::outs() << "\n";
    llvm::outs() << "********** type checking complete **********\n";
    for (const auto decl : ast.declarations) {
      llvm::outs() << *decl;
    }
    llvm::outs() << "********** starting execution **********\n";
  }
  int result =
      Interpreter(arena, trace).InterpProgram(ast.declarations, *ast.main_call);
  llvm::outs() << "result: " << result << "\n";
}

}  // namespace Carbon
