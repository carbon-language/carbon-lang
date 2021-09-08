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
static void AddIntrinsics(std::vector<Ptr<const Declaration>>* declarations) {
  SourceLocation loc("<intrinsic>", 0);
  std::vector<TuplePattern::Field> print_fields = {TuplePattern::Field(
      "0", global_arena->New<BindingPattern>(
               loc, "format_str",
               global_arena->New<ExpressionPattern>(
                   global_arena->New<StringTypeLiteral>(loc))))};
  auto print_return =
      global_arena->New<Return>(loc,
                                global_arena->New<IntrinsicExpression>(
                                    IntrinsicExpression::IntrinsicKind::Print),
                                false);
  auto print = global_arena->New<FunctionDeclaration>(
      global_arena->New<FunctionDefinition>(
          loc, "Print", std::vector<GenericBinding>(),
          global_arena->New<TuplePattern>(loc, print_fields),
          global_arena->New<ExpressionPattern>(
              global_arena->New<TupleLiteral>(loc)),
          /*is_omitted_return_type=*/false, print_return));
  declarations->insert(declarations->begin(), print);
}

void ExecProgram(AST ast) {
  AddIntrinsics(&ast.declarations);
  if (tracing_output) {
    llvm::outs() << "********** source program **********\n";
    for (const auto decl : ast.declarations) {
      llvm::outs() << *decl;
    }
    llvm::outs() << "********** type checking **********\n";
  }
  TypeChecker type_checker;
  TypeChecker::TypeCheckContext p = type_checker.TopLevel(ast.declarations);
  TypeEnv top = p.types;
  Env ct_top = p.values;
  std::vector<Ptr<const Declaration>> new_decls;
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
  int result = Interpreter().InterpProgram(new_decls);
  llvm::outs() << "result: " << result << "\n";
}

}  // namespace Carbon
