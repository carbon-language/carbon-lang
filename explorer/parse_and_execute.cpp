// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/check.h"
#include "common/error.h"
#include "explorer/interpreter/exec_program.h"
#include "explorer/interpreter/trace_stream.h"
#include "explorer/syntax/parse.h"
#include "explorer/syntax/prelude.h"

namespace Carbon::Testing {

auto ParseAndExecute(const std::string& prelude_path, const std::string& source)
    -> ErrorOr<int> {
  Arena arena;
  CARBON_ASSIGN_OR_RETURN(AST ast,
                          ParseFromString(&arena, "test.carbon", source,
                                          /*parser_debug=*/false));

  AddPrelude(prelude_path, &arena, &ast.declarations,
             &ast.num_prelude_declarations);
  TraceStream trace_stream;

  // Use llvm::nulls() to suppress output from the Print intrinsic.
  CARBON_ASSIGN_OR_RETURN(
      ast, AnalyzeProgram(&arena, ast, &trace_stream, &llvm::nulls()));
  return ExecProgram(&arena, ast, &trace_stream, &llvm::nulls());
}

}  // namespace Carbon::Testing
