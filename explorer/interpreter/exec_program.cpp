// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/exec_program.h"

#include <variant>

#include "common/check.h"
#include "common/error.h"
#include "common/ostream.h"
#include "explorer/common/arena.h"
#include "explorer/common/trace_stream.h"
#include "explorer/interpreter/interpreter.h"
#include "explorer/interpreter/resolve_control_flow.h"
#include "explorer/interpreter/resolve_names.h"
#include "explorer/interpreter/resolve_unformed.h"
#include "explorer/interpreter/type_checker.h"
#include "llvm/Support/Error.h"

namespace Carbon {

auto AnalyzeProgram(Nonnull<Arena*> arena, AST ast,
                    Nonnull<TraceStream*> trace_stream,
                    Nonnull<llvm::raw_ostream*> print_stream) -> ErrorOr<AST> {
  SetProgramPhase set_prog_phase(*trace_stream, ProgramPhase::SourceProgram);
  SetFileContext set_file_ctx(*trace_stream, std::nullopt);

  if (trace_stream->is_enabled()) {
    *trace_stream << "********** source program **********\n";
    for (int i = ast.num_prelude_declarations;
         i < static_cast<int>(ast.declarations.size()); ++i) {
      *trace_stream << *ast.declarations[i];
    }
  }

  SourceLocation source_loc("<Main()>", 0, FileKind::Main);
  ast.main_call = arena->New<CallExpression>(
      source_loc, arena->New<IdentifierExpression>(source_loc, "Main"),
      arena->New<TupleLiteral>(source_loc));

  // Although name resolution is currently done once, generic programming
  // (particularly templates) may require more passes.
  set_prog_phase.update_phase(ProgramPhase::NameResolution);
  if (trace_stream->is_enabled()) {
    *trace_stream << "********** resolving names **********\n";
  }
  CARBON_RETURN_IF_ERROR(ResolveNames(ast));

  set_prog_phase.update_phase(ProgramPhase::ControlFlowResolution);
  if (trace_stream->is_enabled()) {
    *trace_stream << "********** resolving control flow **********\n";
  }
  CARBON_RETURN_IF_ERROR(ResolveControlFlow(ast));

  set_prog_phase.update_phase(ProgramPhase::TypeChecking);
  if (trace_stream->is_enabled()) {
    *trace_stream << "********** type checking **********\n";
  }
  CARBON_RETURN_IF_ERROR(
      TypeChecker(arena, trace_stream, print_stream).TypeCheck(ast));

  set_prog_phase.update_phase(ProgramPhase::UnformedVariableResolution);
  if (trace_stream->is_enabled()) {
    *trace_stream << "********** resolving unformed variables **********\n";
  }
  CARBON_RETURN_IF_ERROR(ResolveUnformed(ast));

  set_prog_phase.update_phase(ProgramPhase::Declarations);
  if (trace_stream->is_enabled()) {
    *trace_stream << "********** printing declarations **********\n";
    for (auto& declaration : ast.declarations) {
      set_file_ctx.update_source_loc(declaration->source_loc());
      if (trace_stream->is_enabled()) {
        *trace_stream << *declaration;
      }
    }
  }
  return ast;
}

auto ExecProgram(Nonnull<Arena*> arena, AST ast,
                 Nonnull<TraceStream*> trace_stream,
                 Nonnull<llvm::raw_ostream*> print_stream) -> ErrorOr<int> {
  SetProgramPhase set_program_phase(*trace_stream, ProgramPhase::Execution);
  if (trace_stream->is_enabled()) {
    *trace_stream << "********** starting execution **********\n";
  }
  CARBON_ASSIGN_OR_RETURN(
      auto interpreter_result,
      InterpProgram(ast, arena, trace_stream, print_stream));
  if (trace_stream->is_enabled()) {
    *trace_stream << "interpreter result: " << interpreter_result << "\n";
  }
  return interpreter_result;
}

}  // namespace Carbon
