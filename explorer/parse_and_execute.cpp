// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <locale>

#include "common/check.h"
#include "common/error.h"
#include "explorer/interpreter/exec_program.h"
#include "explorer/interpreter/stack_space.h"
#include "explorer/syntax/parse.h"
#include "explorer/syntax/prelude.h"

namespace Carbon {

static auto ParseAndExecuteHelper(std::function<ErrorOr<AST>(Arena*)> parse,
                                  const std::string& prelude_path,
                                  TraceStream* trace_stream) -> ErrorOr<int> {
  return InitStackSpace<ErrorOr<int>>([&]() -> ErrorOr<int> {
    Arena arena;
    auto time_start = std::chrono::system_clock::now();

    ErrorOr<AST> parse_result = parse(&arena);
    if (!parse_result.ok()) {
      return ErrorBuilder() << "SYNTAX ERROR: " << parse_result.error();
    }

    auto time_after_parse = std::chrono::system_clock::now();

    AddPrelude(prelude_path, &arena, &parse_result->declarations,
               &parse_result->num_prelude_declarations);

    auto time_after_prelude = std::chrono::system_clock::now();

    // Semantically analyze the parsed program.
    ErrorOr<AST> analyze_result =
        AnalyzeProgram(&arena, *parse_result, trace_stream, &llvm::outs());
    if (!analyze_result.ok()) {
      return ErrorBuilder() << "COMPILATION ERROR: " << analyze_result.error();
    }

    auto time_after_analyze = std::chrono::system_clock::now();

    // Run the program.
    ErrorOr<int> exec_result =
        ExecProgram(&arena, *analyze_result, trace_stream, &llvm::outs());

    auto time_after_exec = std::chrono::system_clock::now();
    if (trace_stream->is_enabled()) {
      *trace_stream << "Timings:\n"
                    << "- Parse: "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(
                           time_after_parse - time_start)
                           .count()
                    << "ms\n"
                    << "- AddPrelude: "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(
                           time_after_prelude - time_after_parse)
                           .count()
                    << "ms\n"
                    << "- AnalyzeProgram: "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(
                           time_after_analyze - time_after_prelude)
                           .count()
                    << "ms\n"
                    << "- ExecProgram: "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(
                           time_after_exec - time_after_analyze)
                           .count()
                    << "ms\n";
    }

    if (!exec_result.ok()) {
      return ErrorBuilder() << "RUNTIME ERROR: " << exec_result.error();
    }
    return exec_result;
  });
}

auto ParseAndExecuteFile(const std::string& prelude_path,
                         const std::string& input_file_name, bool parser_debug,
                         TraceStream* trace_stream) -> ErrorOr<int> {
  auto parse = [&](Arena* arena) {
    return Parse(arena, input_file_name, parser_debug);
  };
  return ParseAndExecuteHelper(parse, prelude_path, trace_stream);
}

auto ParseAndExecute(const std::string& prelude_path, const std::string& source)
    -> ErrorOr<int> {
  auto parse = [&](Arena* arena) {
    return ParseFromString(arena, "test.carbon", source,
                           /*parser_debug=*/false);
  };
  TraceStream trace_stream;
  return ParseAndExecuteHelper(parse, prelude_path, &trace_stream);
}

}  // namespace Carbon
