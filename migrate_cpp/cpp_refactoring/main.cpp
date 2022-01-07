// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "migrate_cpp/cpp_refactoring/fn_inserter.h"
#include "migrate_cpp/cpp_refactoring/for_range.h"
#include "migrate_cpp/cpp_refactoring/matcher_manager.h"
#include "migrate_cpp/cpp_refactoring/var_decl.h"

using clang::tooling::RefactoringTool;

// Initialize the files in replacements. Matcher will restrict replacements to
// initialized files.
static void InitReplacements(RefactoringTool* tool) {
  clang::FileManager& files = tool->getFiles();
  Carbon::Matcher::ReplacementMap& repl = tool->getReplacements();
  for (const std::string& path : tool->getSourcePaths()) {
    llvm::ErrorOr<const clang::FileEntry*> file = files.getFile(path);
    if (file.getError()) {
      llvm::report_fatal_error(llvm::Twine("Error accessing `") + path +
                               "`: " + file.getError().message() + "\n");
    }
    repl.insert({files.getCanonicalName(*file).str(), {}});
  }
}

auto main(int argc, const char** argv) -> int {
  llvm::cl::OptionCategory category("C++ refactoring options");
  auto parser =
      clang::tooling::CommonOptionsParser::create(argc, argv, category);
  RefactoringTool tool(parser->getCompilations(), parser->getSourcePathList());
  InitReplacements(&tool);

  // Set up AST matcher callbacks.
  Carbon::MatcherManager matchers(&tool.getReplacements());
  matchers.Register(std::make_unique<Carbon::FnInserterFactory>());
  matchers.Register(std::make_unique<Carbon::ForRangeFactory>());
  matchers.Register(std::make_unique<Carbon::VarDeclFactory>());

  return tool.runAndSave(
      clang::tooling::newFrontendActionFactory(matchers.GetFinder()).get());
}
