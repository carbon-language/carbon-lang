// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/main.h"

#include <unistd.h>

#include <string>

#include "common/error.h"
#include "explorer/base/trace_stream.h"
#include "explorer/parse_and_execute/parse_and_execute.h"
#include "llvm/ADT/ScopeExit.h"
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

  std::string exe =
      llvm::sys::fs::getMainExecutable(argv[0], static_for_main_addr);
  llvm::StringRef install_path = path::parent_path(exe);

  return ExplorerMain(argc, const_cast<const char**>(argv), install_path,
                      relative_prelude_path, llvm::outs(), llvm::errs(),
                      llvm::outs(), *llvm::vfs::getRealFileSystem());
}

auto ExplorerMain(int argc, const char** argv, llvm::StringRef install_path,
                  llvm::StringRef relative_prelude_path,
                  llvm::raw_ostream& out_stream, llvm::raw_ostream& err_stream,
                  llvm::raw_ostream& out_stream_for_trace,
                  llvm::vfs::FileSystem& fs) -> int {
  cl::opt<std::string> input_file_name(cl::Positional, cl::desc("<input file>"),
                                       cl::Required);
  cl::opt<bool> parser_debug("parser_debug",
                             cl::desc("Enable debug output from the parser"));
  cl::opt<std::string> trace_file_name(
      "trace_file",
      cl::desc("Output file for tracing; set to `-` to output to stdout."));

  cl::list<ProgramPhase> trace_phases(
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
                     "Include trace output for all phases.")),
      cl::CommaSeparated);

  enum class TraceFileContext { Main, Prelude, Import, All };
  cl::list<TraceFileContext> trace_file_contexts(
      "trace_file_context",
      cl::desc("Select file contexts for which you want to include the trace "
               "output"),
      cl::values(
          clEnumValN(
              TraceFileContext::Main, "main",
              "Include trace output for file containing the main function"),
          clEnumValN(TraceFileContext::Prelude, "prelude",
                     "Include trace output for prelude"),
          clEnumValN(TraceFileContext::Import, "import",
                     "Include trace output for imports"),
          clEnumValN(TraceFileContext::All, "all",
                     "Include trace output for all files")),
      cl::CommaSeparated);

  CARBON_CHECK(argc > 0);

  // Use the executable path as a base for the relative prelude path.
  llvm::SmallString<256> default_prelude_file(install_path);
  path::append(default_prelude_file,
               path::begin(relative_prelude_path, path::Style::posix),
               path::end(relative_prelude_path));
  std::string default_prelude_file_str(default_prelude_file);
  cl::opt<std::string> prelude_file_name("prelude", cl::desc("<prelude file>"),
                                         cl::init(default_prelude_file_str));

  cl::ParseCommandLineOptions(argc, argv);
  auto reset_parser =
      llvm::make_scope_exit([] { cl::ResetCommandLineParser(); });

  // Set up a stream for trace output.
  std::unique_ptr<llvm::raw_ostream> scoped_trace_stream;
  TraceStream trace_stream;

  if (!trace_file_name.empty()) {
    // Adding allowed phases in the trace_stream.
    trace_stream.set_allowed_phases(trace_phases);

    // Translate --trace_file_context setting into a list of FileKinds.
    llvm::SmallVector<FileKind> trace_file_kinds = {FileKind::Unknown};
    if (!trace_file_contexts.getNumOccurrences()) {
      trace_file_kinds.push_back(FileKind::Main);
    } else {
      for (auto context : trace_file_contexts) {
        switch (context) {
          case TraceFileContext::Main:
            trace_file_kinds.push_back(FileKind::Main);
            break;
          case TraceFileContext::Prelude:
            trace_file_kinds.push_back(FileKind::Prelude);
            break;
          case TraceFileContext::Import:
            trace_file_kinds.push_back(FileKind::Import);
            break;
          case TraceFileContext::All:
            trace_file_kinds.push_back(FileKind::Main);
            trace_file_kinds.push_back(FileKind::Prelude);
            trace_file_kinds.push_back(FileKind::Import);
            break;
        }
      }
    }
    trace_stream.set_allowed_file_kinds(trace_file_kinds);

    if (trace_file_name == "-") {
      trace_stream.set_stream(&out_stream_for_trace);
    } else {
      std::error_code err;
      scoped_trace_stream =
          std::make_unique<llvm::raw_fd_ostream>(trace_file_name, err);
      if (err) {
        err_stream << err.message() << "\n";
        return EXIT_FAILURE;
      }
      trace_stream.set_stream(scoped_trace_stream.get());
    }
  }

  ErrorOr<int> result =
      ParseAndExecute(fs, prelude_file_name, input_file_name, parser_debug,
                      &trace_stream, &out_stream);
  if (result.ok()) {
    // Print the return code to stdout.
    out_stream << "result: " << *result << "\n";
    return EXIT_SUCCESS;
  } else {
    err_stream << result.error() << "\n";
    return EXIT_FAILURE;
  }
}

}  // namespace Carbon
