// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "migrate_cpp/cpp_refactoring/fn_inserter.h"

namespace cam = ::clang::ast_matchers;
namespace ct = ::clang::tooling;

// Initialize the files in replacements. Matcher will restrict replacements to
// initialized files.
static void InitReplacements(ct::RefactoringTool* tool) {
  auto& files = tool->getFiles();
  auto& repl = tool->getReplacements();
  for (const auto& path : tool->getSourcePaths()) {
    auto file = files.getFile(path);
    if (file.getError()) {
      llvm::report_fatal_error("Error accessing `" + path +
                               "`: " + file.getError().message() + "\n");
    }
    repl.insert({files.getCanonicalName(*file).str(), {}});
  }
}

auto main(int argc, const char** argv) -> int {
  llvm::cl::OptionCategory category("C++ refactoring options");
  ct::CommonOptionsParser op(argc, argv, category);
  ct::RefactoringTool tool(op.getCompilations(), op.getSourcePathList());
  InitReplacements(&tool);

  // Set up AST matcher callbacks.
  cam::MatchFinder finder;
  Carbon::FnInserter fn_inserter(tool.getReplacements(), &finder);

  return tool.runAndSave(
      clang::tooling::newFrontendActionFactory(&finder).get());
}
