// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstring>
#include <iostream>

#include "executable_semantics/interpreter/exec_program.h"
#include "executable_semantics/syntax/parse.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

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
  opt<bool> trace_bison_option(
      "trace_bison", llvm::cl::init(true),
      desc("Controls whether bison tracing is enabled when --trace is passed. "
           "Defaults to true."));
  opt<std::string> input_file_name(llvm::cl::Positional, desc("<input file>"),
                                   llvm::cl::Required);

  llvm::cl::ParseCommandLineOptions(argc, argv);

  Carbon::Arena arena;
  std::variant<Carbon::AST, Carbon::SyntaxErrorCode> ast_or_error =
      Carbon::Parse(&arena, input_file_name, trace_option,
                    trace_option && trace_bison_option);

  if (auto* error = std::get_if<Carbon::SyntaxErrorCode>(&ast_or_error)) {
    // Diagnostic already reported to std::cerr; this is just a return code.
    return *error;
  }

  // Typecheck and run the parsed program.
  Carbon::ExecProgram(&arena, std::get<Carbon::AST>(ast_or_error),
                      trace_option);
}
