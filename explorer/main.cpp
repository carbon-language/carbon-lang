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

  cl::list<ProgramPhase> allowed_program_phases(
      "trace_phase",
      cl::desc("Select the program phases to include in the output. By "
               "default, only the execution trace will be added to the trace "
               "output. Use a combination of the following flags to include "
               "outputs for multiple phases:"),
      cl::values(
          clEnumValN(ProgramPhase::SourceProgram, "source_program",
                     "Include trace output for the Source Program phase."),
          clEnumValN(ProgramPhase::NameResolution, "name_resolution",
                     "Include trace output for the Name Resolution phase."),
          clEnumValN(
              ProgramPhase::ControlFlowResolution, "control_flow_resolution",
              "Include trace output for the Control Flow Resolution phase."),
          clEnumValN(ProgramPhase::TypeChecking, "type_checking",
                     "Include trace output for the Type Checking phase."),
          clEnumValN(ProgramPhase::UnformedVariableResolution,
                     "unformed_variables_resolution",
                     "Include trace output for the Unformed Variables "
                     "Resolution phase."),
          clEnumValN(ProgramPhase::Declarations, "declarations",
                     "Include trace output for printing Declarations."),
          clEnumValN(ProgramPhase::Execution, "execution",
                     "Include trace output for Program Execution."),
          clEnumValN(
              ProgramPhase::Timing, "timing",
              "Include timing logs for each phase, indicating the time taken."),
          clEnumValN(ProgramPhase::All, "all",
                     "Include trace output for all phases.")));

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
    // Adding allowed phases in the trace_stream
    trace_stream.set_allowed_phases(allowed_program_phases);
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
    return EXIT_SUCCESS;
  } else {
    llvm::errs() << result.error() << "\n";
    return EXIT_FAILURE;
  }
}

}  // namespace Carbon
