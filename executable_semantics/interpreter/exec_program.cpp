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
      source_loc,
      arena->New<Sequence>(
          source_loc,
          arena->New<Return>(source_loc,
                             arena->New<IntrinsicExpression>(
                                 IntrinsicExpression::Intrinsic::Print),
                             false),
          std::nullopt));
  auto print = arena->New<FunctionDeclaration>(
      source_loc, "Print", std::vector<GenericBinding>(),
      arena->New<TuplePattern>(source_loc, print_params),
      arena->New<ExpressionPattern>(arena->New<TupleLiteral>(source_loc)),
      /*is_omitted_return_type=*/false, print_return);
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
  // Although name resolution is currently done once, generic programming
  // (particularly templates) may require more passes.
  ResolveNames(ast);
  ResolveControlFlow(ast);
  TypeChecker type_checker(arena, trace);
  TypeChecker::TypeCheckContext p = type_checker.TopLevel(&ast.declarations);
  TypeEnv top = p.types;
  Env ct_top = p.values;
  for (const auto decl : ast.declarations) {
    type_checker.TypeCheck(decl, top, ct_top);
  }
  if (trace) {
    llvm::outs() << "\n";
    llvm::outs() << "********** type checking complete **********\n";
    for (const auto decl : ast.declarations) {
      llvm::outs() << *decl;
    }
    llvm::outs() << "********** starting execution **********\n";
  }

  SourceLocation source_loc("<main()>", 0);
  Nonnull<Expression*> call_main = arena->New<CallExpression>(
      source_loc, arena->New<IdentifierExpression>(source_loc, "main"),
      arena->New<TupleLiteral>(source_loc));
  int result =
      Interpreter(arena, trace).InterpProgram(ast.declarations, call_main);
  llvm::outs() << "result: " << result << "\n";
}

}  // namespace Carbon
