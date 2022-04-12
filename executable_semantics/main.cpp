// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <optional>
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
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

namespace cl = llvm::cl;

auto Main(int argc, char* argv[]) -> ErrorOr<Success> {
  llvm::setBugReportMsg(
      "Please report issues to "
      "https://github.com/carbon-language/carbon-lang/issues and include the "
      "crash backtrace.\n");
  llvm::InitLLVM init_llvm(argc, argv);

  // Printing to stderr should flush stdout. This is most noticeable when stderr
  // is piped to stdout.
  llvm::errs().tie(&llvm::outs());

  cl::opt<std::string> input_file_name(cl::Positional, cl::desc("<input file>"),
                                       cl::Required);
  cl::opt<bool> parser_debug("parser_debug",
                             cl::desc("Enable debug output from the parser"));
  cl::opt<std::string> trace_file_name(
      "trace_file",
      cl::desc("Output file for tracing; set to `-` to output to stdout."));
  cl::opt<std::string> prelude_file_name(
      "prelude", cl::desc("<prelude file>"),
      cl::init("executable_semantics/data/prelude.carbon"));

  cl::ParseCommandLineOptions(argc, argv);

  // Set up a stream for trace output.
  std::unique_ptr<llvm::raw_ostream> scoped_trace_stream;
  std::optional<Nonnull<llvm::raw_ostream*>> trace_stream;
  if (!trace_file_name.empty()) {
    if (trace_file_name == "-") {
      trace_stream = &llvm::outs();
    } else {
      std::error_code err;
      scoped_trace_stream =
          std::make_unique<llvm::raw_fd_ostream>(trace_file_name, err);
      if (err) {
        return Error(err.message());
      }
      trace_stream = scoped_trace_stream.get();
    }
  }

  Arena arena;
  ASSIGN_OR_RETURN(AST ast, Parse(&arena, input_file_name, parser_debug));
  AddPrelude(prelude_file_name, &arena, &ast.declarations);

  // Typecheck and run the parsed program.
  ASSIGN_OR_RETURN(int return_code, ExecProgram(&arena, ast, trace_stream));
  // Print the return code to stdout even when we aren't tracing.
  (trace_stream == nullptr ? llvm::outs() : **trace_stream)
      << "result: " << return_code << "\n";
  return Success();
}

}  // namespace Carbon

auto main(int argc, char* argv[]) -> int {
  if (auto result = Carbon::Main(argc, argv); !result.ok()) {
    llvm::errs() << result.error().message() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
