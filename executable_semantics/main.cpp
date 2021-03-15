// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstring>
#include <iostream>

#include "executable_semantics/syntax/parse.h"
#include "executable_semantics/syntax/syntax_helpers.h"
#include "executable_semantics/tracing_flag.h"
#include "llvm/Support/CommandLine.h"

auto main(int argc, char* argv[]) -> int {
  // yydebug = 1;

  using llvm::cl::desc;
  using llvm::cl::opt;
  opt<bool> quiet_option("quiet", desc("Disable tracing"));
  opt<std::string> input_filename(llvm::cl::Positional, desc("<input file>"),
                                  llvm::cl::Required);

  llvm::cl::ParseCommandLineOptions(argc, argv);
  if (quiet_option) {
    Carbon::tracing_output = false;
  }

  std::variant<Carbon::AST, Carbon::SyntaxErrorCode> ast_or_error =
      Carbon::Parse(input_filename);

  if (auto* error = std::get_if<Carbon::SyntaxErrorCode>(&ast_or_error)) {
    // Diagnostic already reported to std::cerr; this is just a return code.
    return *error;
  }

  // Typecheck and run the parsed program.
  Carbon::ExecProgram(std::get<Carbon::AST>(ast_or_error));
}
