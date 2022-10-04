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
#include "explorer/interpreter/resolve_unformed.h"
#include "explorer/interpreter/type_checker.h"
#include "llvm/Support/Error.h"

namespace Carbon {

auto AnalyzeProgram(Nonnull<Arena*> arena, AST ast,
                    std::optional<Nonnull<llvm::raw_ostream*>> trace_stream)
    -> ErrorOr<AST> {
  if (trace_stream) {
    **trace_stream << "********** source program **********\n";
    for (const auto decl : ast.declarations) {
      **trace_stream << *decl;
    }
  }
  SourceLocation source_loc("<Main()>", 0);
  ast.main_call = arena->New<CallExpression>(
      source_loc, arena->New<IdentifierExpression>(source_loc, "Main"),
      arena->New<TupleLiteral>(source_loc));
  // Although name resolution is currently done once, generic programming
  // (particularly templates) may require more passes.
  if (trace_stream) {
    **trace_stream << "********** resolving names **********\n";
  }
  CARBON_RETURN_IF_ERROR(ResolveNames(ast));
  if (trace_stream) {
    **trace_stream << "********** resolving control flow **********\n";
  }
  CARBON_RETURN_IF_ERROR(ResolveControlFlow(ast));
  if (trace_stream) {
    **trace_stream << "********** type checking **********\n";
  }
  CARBON_RETURN_IF_ERROR(TypeChecker(arena, trace_stream).TypeCheck(ast));
  if (trace_stream) {
    **trace_stream << "********** resolving unformed variables **********\n";
  }
  CARBON_RETURN_IF_ERROR(ResolveUnformed(ast));
  if (trace_stream) {
    **trace_stream << "********** printing declarations **********\n";
    for (const auto decl : ast.declarations) {
      **trace_stream << *decl;
    }
  }
  return ast;
}

auto ExecProgram(Nonnull<Arena*> arena, AST ast,
                 std::optional<Nonnull<llvm::raw_ostream*>> trace_stream)
    -> ErrorOr<int> {
  if (trace_stream) {
    **trace_stream << "********** starting execution **********\n";
  }
  return InterpProgram(ast, arena, trace_stream);
}

}  // namespace Carbon
