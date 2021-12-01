// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/nonnull.h"
#include "executable_semantics/interpreter/exec_program.h"
#include "executable_semantics/syntax/parse.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

// The Carbon prelude.
//
// TODO: Make this a separate source file that's embedded in the interpreter
// at build time. See https://github.com/bazelbuild/rules_cc/issues/41 for a
// possible mechanism.
static constexpr std::string_view Prelude = R"(
package Carbon api;

// Note that Print is experimental, and not part of an accepted proposal, but
// is included here for printing state in tests.
fn Print(format_str: String) {
  __intrinsic_print(format_str);
}
)";

// Adds the Carbon prelude to `declarations`.
static void AddPrelude(
    Carbon::Nonnull<Carbon::Arena*> arena,
    std::vector<Carbon::Nonnull<Carbon::Declaration*>>* declarations) {
  std::variant<Carbon::AST, Carbon::SyntaxErrorCode> parse_result =
      ParseFromString(arena, "<prelude>", Prelude, false);
  if (std::holds_alternative<Carbon::SyntaxErrorCode>(parse_result)) {
    // Try again with tracing, to help diagnose the problem.
    ParseFromString(arena, "<prelude>", Prelude, true);
    FATAL() << "Failed to parse prelude.";
  }
  const auto& prelude = std::get<Carbon::AST>(parse_result);
  declarations->insert(declarations->begin(), prelude.declarations.begin(),
                       prelude.declarations.end());
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

  llvm::cl::ParseCommandLineOptions(argc, argv);

  Carbon::Arena arena;
  std::variant<Carbon::AST, Carbon::SyntaxErrorCode> ast_or_error =
      Carbon::Parse(&arena, input_file_name, trace_option);

  if (auto* error = std::get_if<Carbon::SyntaxErrorCode>(&ast_or_error)) {
    // Diagnostic already reported to std::cerr; this is just a return code.
    return *error;
  }
  auto& ast = std::get<Carbon::AST>(ast_or_error);

  AddPrelude(&arena, &ast.declarations);

  // Typecheck and run the parsed program.
  Carbon::ExecProgram(&arena, std::get<Carbon::AST>(ast_or_error),
                      trace_option);
}
