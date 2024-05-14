// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/exec_program.h"

#include <variant>

#include "common/check.h"
#include "common/error.h"
#include "common/ostream.h"
#include "explorer/base/arena.h"
#include "explorer/base/trace_stream.h"
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
    trace_stream->Heading("source program");
    llvm::ListSeparator sep("\n\n");
    for (auto& declaration : ast.declarations) {
      set_file_ctx.update_source_loc(declaration->source_loc());
      if (trace_stream->is_enabled()) {
        *trace_stream << sep << *declaration;
      }
    }
    if (trace_stream->is_enabled()) {
      *trace_stream << "\n";
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
    trace_stream->Heading("resolving names");
  }
  CARBON_RETURN_IF_ERROR(ResolveNames(ast, trace_stream));

  set_prog_phase.update_phase(ProgramPhase::ControlFlowResolution);
  if (trace_stream->is_enabled()) {
    trace_stream->Heading("resolving control flow");
  }
  CARBON_RETURN_IF_ERROR(ResolveControlFlow(trace_stream, ast));

  set_prog_phase.update_phase(ProgramPhase::TypeChecking);
  if (trace_stream->is_enabled()) {
    trace_stream->Heading("type checking");
  }
  CARBON_RETURN_IF_ERROR(
      TypeChecker(arena, trace_stream, print_stream).TypeCheck(ast));

  set_prog_phase.update_phase(ProgramPhase::UnformedVariableResolution);
  if (trace_stream->is_enabled()) {
    trace_stream->Heading("resolving unformed variables");
  }
  CARBON_RETURN_IF_ERROR(ResolveUnformed(trace_stream, ast));

  set_prog_phase.update_phase(ProgramPhase::Declarations);
  if (trace_stream->is_enabled()) {
    trace_stream->Heading("printing declarations");
    llvm::ListSeparator sep("\n\n");
    for (auto& declaration : ast.declarations) {
      set_file_ctx.update_source_loc(declaration->source_loc());
      if (trace_stream->is_enabled()) {
        *trace_stream << sep << *declaration;
      }
    }
    if (trace_stream->is_enabled()) {
      *trace_stream << "\n";
    }
  }
  return ast;
}

auto ExecProgram(Nonnull<Arena*> arena, AST ast,
                 Nonnull<TraceStream*> trace_stream,
                 Nonnull<llvm::raw_ostream*> print_stream) -> ErrorOr<int> {
  SetProgramPhase set_program_phase(*trace_stream, ProgramPhase::Execution);
  if (trace_stream->is_enabled()) {
    trace_stream->Heading("starting execution");
  }
  CARBON_ASSIGN_OR_RETURN(
      auto interpreter_result,
      InterpProgram(ast, arena, trace_stream, print_stream));
  if (trace_stream->is_enabled()) {
    trace_stream->Result() << "interpreter result: " << interpreter_result
                           << "\n";
  }
  return interpreter_result;
}

}  // namespace Carbon
