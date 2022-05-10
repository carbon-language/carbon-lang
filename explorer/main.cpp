// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "common/error.h"
#include "explorer/common/arena.h"
#include "explorer/common/nonnull.h"
#include "explorer/interpreter/exec_program.h"
#include "explorer/syntax/parse.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

// Adds the Carbon prelude to `declarations`.
static void AddPrelude(
    std::string_view prelude_file_name, Carbon::Nonnull<Carbon::Arena*> arena,
    std::vector<Carbon::Nonnull<Carbon::Declaration*>>* declarations) {
  Carbon::ErrorOr<Carbon::AST> parse_result =
      Carbon::Parse(arena, prelude_file_name, false);
  if (!parse_result.ok()) {
    // Try again with tracing, to help diagnose the problem.
    Carbon::ErrorOr<Carbon::AST> trace_parse_result =
        Carbon::Parse(arena, prelude_file_name, true);
    FATAL() << "Failed to parse prelude: "
            << trace_parse_result.error().message();
  }
  const auto& prelude = *parse_result;
  declarations->insert(declarations->begin(), prelude.declarations.begin(),
                       prelude.declarations.end());
}

// Prints an error message and returns error code value.
auto PrintError(const Carbon::Error& error) -> int {
  llvm::errs() << error.message() << "\n";
  return EXIT_FAILURE;
}

auto main(int argc, char* argv[]) -> int {
  llvm::setBugReportMsg(
      "Please report issues to "
      "https://github.com/carbon-language/carbon-lang/issues and include the "
      "crash backtrace.\n");
  llvm::InitLLVM init_llvm(argc, argv);

  // Printing to stderr should flush stdout. This is most noticeable when stderr
  // is piped to stdout.
  llvm::errs().tie(&llvm::outs());

  using llvm::cl::desc;
  using llvm::cl::opt;
  opt<bool> trace_option("trace", desc("Enable tracing"));
  opt<std::string> input_file_name(llvm::cl::Positional, desc("<input file>"),
                                   llvm::cl::Required);
  opt<std::string> prelude_file_name(
      "prelude", desc("<prelude file>"),
      llvm::cl::init("explorer/data/prelude.carbon"));

  llvm::cl::ParseCommandLineOptions(argc, argv);

  Carbon::Arena arena;
  Carbon::ErrorOr<Carbon::AST> ast =
      Carbon::Parse(&arena, input_file_name, trace_option);
  if (!ast.ok()) {
    return PrintError(ast.error());
  }
  AddPrelude(prelude_file_name, &arena, &ast->declarations);

  // Typecheck and run the parsed program.
  Carbon::ErrorOr<int> result = Carbon::ExecProgram(&arena, *ast, trace_option);
  if (!result.ok()) {
    return PrintError(result.error());
  }
}
