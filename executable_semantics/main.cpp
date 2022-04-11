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
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

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

  // Find the path of the executable if possible and use that as a relative root
  // for finding the prelude. FIXME: Currently, this assumes a Bazel-like
  // runfiles tree rather than any kind of installation tree.
  llvm::SmallString<256> exe_path(argv[0]);
  if (!llvm::sys::fs::exists(exe_path)) {
    // Try to lookup the program name in the `PATH`.
    if (llvm::ErrorOr<std::string> path =
            llvm::sys::findProgramByName(exe_path)) {
      exe_path = *path;
    }
  }
  // If we have a valid path, find the parent directory. Otherwise, just use an
  // empty string to get the current working directory.
  if (llvm::sys::fs::exists(exe_path)) {
    llvm::sys::path::remove_filename(exe_path);
  } else {
    exe_path = "";
  }
  llvm::SmallString<256> prelude_path = exe_path;
  llvm::sys::path::append(prelude_path, "data/prelude.carbon");
  opt<std::string> prelude_file_name("prelude", desc("<prelude file>"),
                                     llvm::cl::init(prelude_path.str().str()));

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
