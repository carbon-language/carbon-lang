// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/main.h"

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "common/error.h"
#include "explorer/common/trace_stream.h"
#include "explorer/parse_and_execute/parse_and_execute.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

namespace cl = llvm::cl;
namespace path = llvm::sys::path;

auto ExplorerMain(int argc, char** argv, void* static_for_main_addr,
                  llvm::StringRef relative_prelude_path) -> int {
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

  // Use the executable path as a base for the relative prelude path.
  std::string exe =
      llvm::sys::fs::getMainExecutable(argv[0], static_for_main_addr);
  llvm::StringRef install_path = path::parent_path(exe);
  llvm::SmallString<256> default_prelude_file(install_path);
  path::append(default_prelude_file,
               path::begin(relative_prelude_path, path::Style::posix),
               path::end(relative_prelude_path));
  std::string default_prelude_file_str(default_prelude_file);
  cl::opt<std::string> prelude_file_name("prelude", cl::desc("<prelude file>"),
                                         cl::init(default_prelude_file_str));

  cl::ParseCommandLineOptions(argc, argv);

  // Set up a stream for trace output.
  std::unique_ptr<llvm::raw_ostream> scoped_trace_stream;
  TraceStream trace_stream;
  if (!trace_file_name.empty()) {
    if (trace_file_name == "-") {
      trace_stream.set_stream(&llvm::outs());
    } else {
      std::error_code err;
      scoped_trace_stream =
          std::make_unique<llvm::raw_fd_ostream>(trace_file_name, err);
      if (err) {
        llvm::errs() << err.message() << "\n";
        return EXIT_FAILURE;
      }
      trace_stream.set_stream(scoped_trace_stream.get());
    }
  }

  ErrorOr<int> result =
      ParseAndExecuteFile(prelude_file_name, input_file_name, parser_debug,
                          &trace_stream, &llvm::outs());
  if (result.ok()) {
    // Print the return code to stdout.
    llvm::outs() << "result: " << *result << "\n";

    // When there's a dedicated trace file, print the return code to it too.
    if (scoped_trace_stream) {
      trace_stream << "result: " << *result << "\n";
    }
    return EXIT_SUCCESS;
  } else {
    llvm::errs() << result.error() << "\n";
    return EXIT_FAILURE;
  }
}

}  // namespace Carbon
