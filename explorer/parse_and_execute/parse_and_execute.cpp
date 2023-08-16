// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/parse_and_execute/parse_and_execute.h"

#include <locale>

#include "common/error.h"
#include "explorer/base/trace_stream.h"
#include "explorer/interpreter/exec_program.h"
#include "explorer/interpreter/stack_space.h"
#include "explorer/syntax/parse.h"
#include "explorer/syntax/prelude.h"
#include "llvm/ADT/ScopeExit.h"

namespace Carbon {

// Returns a scope exit function for printing the timing of a step on scope
// exit. Note the use prints step timings in reverse order.
static auto PrintTimingOnExit(TraceStream* trace_stream, const char* label,
                              std::chrono::steady_clock::time_point* cursor) {
  auto end = std::chrono::steady_clock::now();
  auto duration = end - *cursor;
  *cursor = end;
  auto exit_scope_function = llvm::make_scope_exit([=]() {
    SetProgramPhase set_program_phase(*trace_stream, ProgramPhase::Timing);
    if (trace_stream->is_enabled()) {
      *trace_stream << "Time elapsed in " << label << ": "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(
                           duration)
                           .count()
                    << "ms\n";
    }
  });
  return exit_scope_function;
}

auto ParseAndExecute(llvm::vfs::FileSystem& fs, std::string_view prelude_path,
                     std::string_view input_file_name, bool parser_debug,
                     Nonnull<TraceStream*> trace_stream,
                     Nonnull<llvm::raw_ostream*> print_stream) -> ErrorOr<int> {
  return RunWithExtraStack([&]() -> ErrorOr<int> {
    Arena arena;
    auto cursor = std::chrono::steady_clock::now();

    ErrorOr<AST> parse_result =
        Parse(fs, &arena, input_file_name, FileKind::Main, parser_debug);
    auto print_parse_time = PrintTimingOnExit(trace_stream, "Parse", &cursor);
    if (!parse_result.ok()) {
      return ErrorBuilder() << "SYNTAX ERROR: " << parse_result.error();
    }

    AddPrelude(fs, prelude_path, &arena, &parse_result->declarations,
               &parse_result->num_prelude_declarations);
    auto print_prelude_time =
        PrintTimingOnExit(trace_stream, "AddPrelude", &cursor);

    // Semantically analyze the parsed program.
    ErrorOr<AST> analyze_result =
        AnalyzeProgram(&arena, *parse_result, trace_stream, print_stream);
    auto print_analyze_time =
        PrintTimingOnExit(trace_stream, "AnalyzeProgram", &cursor);
    if (!analyze_result.ok()) {
      return ErrorBuilder() << "COMPILATION ERROR: " << analyze_result.error();
    }

    // Run the program.
    ErrorOr<int> exec_result =
        ExecProgram(&arena, *analyze_result, trace_stream, print_stream);
    auto print_exec_time =
        PrintTimingOnExit(trace_stream, "ExecProgram", &cursor);

    if (!exec_result.ok()) {
      return ErrorBuilder() << "RUNTIME ERROR: " << exec_result.error();
    }

    auto print_trace_timing_heading = llvm::make_scope_exit([=]() {
      SetProgramPhase set_prog_phase(*trace_stream, ProgramPhase::Timing);
      if (trace_stream->is_enabled()) {
        trace_stream->Heading("printing timing");
      }
    });

    return exec_result;
  });
}

}  // namespace Carbon
