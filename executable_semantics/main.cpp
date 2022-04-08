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
#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/nonnull.h"
#include "executable_semantics/interpreter/exec_program.h"
#include "executable_semantics/syntax/parse.h"
#include "executable_semantics/syntax/prelude.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

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
      llvm::cl::init("executable_semantics/data/prelude.carbon"));

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
