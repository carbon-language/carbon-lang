// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/main.h"

#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "common/error.h"
#include "explorer/common/arena.h"
#include "explorer/common/nonnull.h"
#include "explorer/interpreter/exec_program.h"
#include "explorer/syntax/parse.h"
#include "explorer/syntax/prelude.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

namespace cl = llvm::cl;

static auto Main(llvm::StringRef default_prelude_file, int argc, char* argv[])
    -> bool {
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

  // Find the path of the executable if possible and use that as a relative root
  cl::opt<std::string> prelude_file_name("prelude", cl::desc("<prelude file>"),
                                         cl::init(default_prelude_file.str()));
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
        llvm::errs() << err.message() << "\n";
        return false;
      }
      trace_stream = scoped_trace_stream.get();
    }
  }

  Arena arena;
  AST ast;
  if (ErrorOr<AST> parse_result = Parse(&arena, input_file_name, parser_debug);
      parse_result.ok()) {
    ast = *std::move(parse_result);
  } else {
    llvm::errs() << "SYNTAX ERROR: " << parse_result.error() << "\n";
    return false;
  }

  AddPrelude(prelude_file_name, &arena, &ast.declarations);

  // Semantically analyze the parsed program.
  if (ErrorOr<AST> analyze_result = AnalyzeProgram(&arena, ast, trace_stream);
      analyze_result.ok()) {
    ast = *std::move(analyze_result);
  } else {
    llvm::errs() << "COMPILATION ERROR: " << analyze_result.error() << "\n";
    return false;
  }

  // Run the program.
  if (ErrorOr<int> exec_result = ExecProgram(&arena, ast, trace_stream);
      exec_result.ok()) {
    // Print the return code to stdout.
    llvm::outs() << "result: " << *exec_result << "\n";

    // When there's a dedicated trace file, print the return code to it too.
    if (scoped_trace_stream) {
      **trace_stream << "result: " << *exec_result << "\n";
    }
  } else {
    llvm::errs() << "RUNTIME ERROR: " << exec_result.error() << "\n";
    return false;
  }

  return true;
}

auto ExplorerMain(llvm::StringRef default_prelude_file, int argc, char** argv)
    -> int {
  return Main(default_prelude_file, argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace Carbon
