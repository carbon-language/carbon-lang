// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/main.h"

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

namespace Carbon {

namespace cl = llvm::cl;

static auto Main(llvm::StringRef default_prelude_file, int argc, char* argv[])
    -> ErrorOr<Success> {
  llvm::setBugReportMsg(
      "Please report issues to "
      "https://github.com/carbon-language/carbon-lang/issues and include the "
      "crash backtrace.\n");
  llvm::InitLLVM init_llvm(argc, argv);

  // Printing to stderr should flush stdout. This is most noticeable when stderr
  // is piped to stdout.
  llvm::errs().tie(&llvm::outs());

  cl::opt<bool> trace_option("trace", cl::desc("Enable tracing"));
  cl::opt<std::string> input_file_name(cl::Positional, cl::desc("<input file>"),
                                       cl::Required);

  // Find the path of the executable if possible and use that as a relative root
  cl::opt<std::string> prelude_file_name("prelude", cl::desc("<prelude file>"),
                                         cl::init(default_prelude_file.str()));
  cl::ParseCommandLineOptions(argc, argv);

  Arena arena;
  ASSIGN_OR_RETURN(AST ast, Parse(&arena, input_file_name, trace_option));
  AddPrelude(prelude_file_name, &arena, &ast.declarations);

  // Typecheck and run the parsed program.
  ASSIGN_OR_RETURN(int unused_return_code,
                   ExecProgram(&arena, ast, trace_option));
  (void)unused_return_code;
  return Success();
}

auto ExecutableSemanticsMain(llvm::StringRef default_prelude_file, int argc,
                             char** argv) -> int {
  if (auto result = Main(default_prelude_file, argc, argv); !result.ok()) {
    llvm::errs() << result.error().message() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}  // namespace Carbon
