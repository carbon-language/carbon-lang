// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/syntax_helpers.h"

#include "common/check.h"
#include "common/ostream.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/tracing_flag.h"
#include "executable_semantics/interpreter/interpreter.h"
#include "executable_semantics/interpreter/typecheck.h"

namespace Carbon {

// Adds builtins, currently only Print(). Note Print() is experimental, not
// standardized, but is made available for printing state in tests.
static void AddIntrinsics(std::list<Ptr<const Declaration>>* fs) {
  SourceLocation loc = {.filename = "<intrinsic>", .line_num = -1};
  std::vector<TuplePattern::Field> print_fields = {TuplePattern::Field(
      "0", global_arena->RawNew<BindingPattern>(
               loc, "format_str",
               global_arena->RawNew<ExpressionPattern>(
                   global_arena->RawNew<StringTypeLiteral>(loc))))};
  auto* print_return = global_arena->RawNew<Return>(
      loc,
      global_arena->RawNew<IntrinsicExpression>(
          IntrinsicExpression::IntrinsicKind::Print),
      false);
  auto print = global_arena->New<FunctionDeclaration>(
      global_arena->RawNew<FunctionDefinition>(
          loc, "Print", std::vector<GenericBinding>(),
          global_arena->RawNew<TuplePattern>(loc, print_fields),
          global_arena->RawNew<ExpressionPattern>(
              global_arena->RawNew<TupleLiteral>(loc)),
          /*is_omitted_return_type=*/false, print_return));
  fs->insert(fs->begin(), print);
}

void ExecProgram(std::list<Ptr<const Declaration>> fs) {
  AddIntrinsics(&fs);
  if (tracing_output) {
    llvm::outs() << "********** source program **********\n";
    for (const auto decl : fs) {
      llvm::outs() << *decl;
    }
    llvm::outs() << "********** type checking **********\n";
  }
  state = global_arena->RawNew<State>();  // Compile-time state.
  TypeCheckContext p = TopLevel(fs);
  TypeEnv top = p.types;
  Env ct_top = p.values;
  std::list<Ptr<const Declaration>> new_decls;
  for (const auto decl : fs) {
    new_decls.push_back(MakeTypeChecked(decl, top, ct_top));
  }
  if (tracing_output) {
    llvm::outs() << "\n";
    llvm::outs() << "********** type checking complete **********\n";
    for (const auto decl : new_decls) {
      llvm::outs() << *decl;
    }
    llvm::outs() << "********** starting execution **********\n";
  }
  int result = InterpProgram(new_decls);
  llvm::outs() << "result: " << result << "\n";
}

}  // namespace Carbon
