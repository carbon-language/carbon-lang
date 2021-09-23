// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/exec_program.h"

#include "common/check.h"
#include "common/ostream.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/tracing_flag.h"
#include "executable_semantics/interpreter/interpreter.h"
#include "executable_semantics/interpreter/type_checker.h"

namespace Carbon {

// Adds builtins, currently only Print(). Note Print() is experimental, not
// standardized, but is made available for printing state in tests.
static void AddIntrinsics(Nonnull<Arena*> arena,
                          std::vector<Nonnull<Declaration*>>* declarations) {
  SourceLocation loc("<intrinsic>", 0);
  std::vector<TuplePattern::Field> print_fields = {TuplePattern::Field(
      "0",
      arena->New<BindingPattern>(
          loc, "format_str",
          arena->New<ExpressionPattern>(arena->New<StringTypeLiteral>(loc))))};
  auto print_return =
      arena->New<Return>(loc,
                         arena->New<IntrinsicExpression>(
                             IntrinsicExpression::IntrinsicKind::Print),
                         false);
  auto print = arena->New<FunctionDeclaration>(arena->New<FunctionDefinition>(
      loc, "Print", std::vector<GenericBinding>(),
      arena->New<TuplePattern>(loc, print_fields),
      arena->New<ExpressionPattern>(arena->New<TupleLiteral>(loc)),
      /*is_omitted_return_type=*/false, print_return));
  declarations->insert(declarations->begin(), print);
}

void ExecProgram(Nonnull<Arena*> arena, AST ast) {
  AddIntrinsics(arena, &ast.declarations);
  if (tracing_output) {
    llvm::outs() << "********** source program **********\n";
    for (const auto decl : ast.declarations) {
      llvm::outs() << *decl;
    }
    llvm::outs() << "********** type checking **********\n";
  }
  TypeChecker type_checker(arena);
  TypeChecker::TypeCheckContext p = type_checker.TopLevel(&ast.declarations);
  TypeEnv top = p.types;
  Env ct_top = p.values;
  std::vector<Nonnull<const Declaration*>> new_decls;
  for (const auto decl : ast.declarations) {
    new_decls.push_back(type_checker.MakeTypeChecked(decl, top, ct_top));
  }
  if (tracing_output) {
    llvm::outs() << "\n";
    llvm::outs() << "********** type checking complete **********\n";
    for (const auto decl : new_decls) {
      llvm::outs() << *decl;
    }
    llvm::outs() << "********** starting execution **********\n";
  }

  SourceLocation loc("<main()>", 0);
  Nonnull<Expression*> call_main = arena->New<CallExpression>(
      loc, arena->New<IdentifierExpression>(loc, "main"),
      arena->New<TupleLiteral>(loc));
  int result = Interpreter(arena).InterpProgram(new_decls, call_main);
  llvm::outs() << "result: " << result << "\n";
}

}  // namespace Carbon
