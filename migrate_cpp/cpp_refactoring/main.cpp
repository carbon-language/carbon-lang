// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"

using clang::IfStmt;
using clang::Stmt;
using clang::ast_matchers::MatchFinder;
using clang::tooling::Replacement;
using clang::tooling::Replacements;

class IfStmtHandler : public MatchFinder::MatchCallback {
 public:
  IfStmtHandler(std::map<std::string, Replacements>* in_replacements)
      : replacements(in_replacements) {}

  virtual void run(const MatchFinder::MatchResult& Result) {
    // The matched 'if' statement was bound to 'ifStmt'.
    if (const IfStmt* IfS = Result.Nodes.getNodeAs<clang::IfStmt>("ifStmt")) {
      const Stmt* Then = IfS->getThen();
      Replacement Rep(*(Result.SourceManager), Then->getBeginLoc(), 0,
                      "// the 'if' part\n");
      auto err = (*replacements)[std::string(Rep.getFilePath())].add(Rep);
      if (err) {
        llvm::errs() << err;
      }

      if (const Stmt* Else = IfS->getElse()) {
        Replacement Rep(*(Result.SourceManager), Else->getBeginLoc(), 0,
                        "// the 'else' part\n");
        auto err = (*replacements)[std::string(Rep.getFilePath())].add(Rep);
        if (err) {
          llvm::errs() << err;
        }
      }
    }
  }

 private:
  std::map<std::string, Replacements>* replacements;
};

int main(int argc, const char** argv) {
  llvm::cl::OptionCategory category("C++ refactoring options");
  clang::tooling::CommonOptionsParser op(argc, argv, category);
  clang::tooling::RefactoringTool tool(op.getCompilations(),
                                       op.getSourcePathList());

  // Set up AST matcher callbacks.
  IfStmtHandler HandlerForIf(&tool.getReplacements());

  MatchFinder Finder;
  Finder.addMatcher(clang::ast_matchers::ifStmt().bind("ifStmt"),
                    &HandlerForIf);

  return tool.runAndSave(
      clang::tooling::newFrontendActionFactory(&Finder).get());
}
