// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"

namespace cam = ::clang::ast_matchers;
namespace ct = ::clang::tooling;

class Matcher : public cam::MatchFinder::MatchCallback {
 public:
  explicit Matcher(std::map<std::string, ct::Replacements>* in_replacements)
      : replacements(in_replacements) {}

  virtual ~Matcher() {}

  void AddReplacement(const clang::SourceManager& sm,
                      clang::CharSourceRange range,
                      llvm::StringRef replacement_text) {
    if (!range.isValid()) {
      llvm::errs() << "Invalid range: " << range.getAsRange().printToString(sm)
                   << "\n";
      return;
    }
    if (sm.getDecomposedLoc(range.getBegin()).first !=
        sm.getDecomposedLoc(range.getEnd()).first) {
      llvm::errs() << "Range spans macro expansions: "
                   << range.getAsRange().printToString(sm) << "\n";
      return;
    }
    if (sm.getFileID(range.getBegin()) != sm.getFileID(range.getEnd())) {
      llvm::errs() << "Range spans files: "
                   << range.getAsRange().printToString(sm) << "\n";
      return;
    }

    auto rep =
        ct::Replacement(sm, sm.getExpansionRange(range), replacement_text);
    auto err = (*replacements)[std::string(rep.getFilePath())].add(rep);
    if (err) {
      llvm::errs() << "Error with replacement `" << rep.toString()
                   << "`: " << err << "\n";
      exit(1);
    }
  }

 private:
  std::map<std::string, ct::Replacements>* replacements;
};

class FnInserter : public Matcher {
 public:
  explicit FnInserter(std::map<std::string, ct::Replacements>* in_replacements,
                      cam::MatchFinder* finder)
      : Matcher(in_replacements) {
    finder->addMatcher(cam::functionDecl(cam::isExpansionInMainFile(),
                                         cam::hasTrailingReturn())
                           .bind(Label),
                       this);
  }

  void run(const cam::MatchFinder::MatchResult& result) override {
    // The matched 'if' statement was bound to 'ifStmt'.
    const auto* decl = result.Nodes.getNodeAs<clang::FunctionDecl>(Label);
    if (!decl) {
      llvm::errs() << "getNodeAs failed for " << Label;
      exit(1);
    }
    auto begin = decl->getBeginLoc();
    // Replace the first token in the range, `auto`.
    auto range = clang::CharSourceRange::getTokenRange(begin, begin);
    AddReplacement(*(result.SourceManager), range, "fn");
  }

 private:
  static constexpr char Label[] = "FnInserter";
};

auto main(int argc, const char** argv) -> int {
  llvm::cl::OptionCategory category("C++ refactoring options");
  clang::tooling::CommonOptionsParser op(argc, argv, category);
  clang::tooling::RefactoringTool tool(op.getCompilations(),
                                       op.getSourcePathList());

  // Set up AST matcher callbacks.
  cam::MatchFinder finder;
  FnInserter fn_inserter(&tool.getReplacements(), &finder);

  return tool.runAndSave(
      clang::tooling::newFrontendActionFactory(&finder).get());
}
