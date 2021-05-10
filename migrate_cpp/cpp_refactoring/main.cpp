// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "clang/Tooling/Refactoring.h"
#include "clang/lib/ASTMatchers/ASTMatchFinder.h"

using clang::MatchFinder;

class IfStmtHandler : public MatchFinder::MatchCallback {
 public:
  IfStmtHandler(Replacements* Replace) : Replace(Replace) {}

  virtual void run(const MatchFinder::MatchResult& Result) {
    // The matched 'if' statement was bound to 'ifStmt'.
    if (const IfStmt* IfS = Result.Nodes.getNodeAs<clang::IfStmt>("ifStmt")) {
      const Stmt* Then = IfS->getThen();
      Replacement Rep(*(Result.SourceManager), Then->getLocStart(), 0,
                      "// the 'if' part\n");
      Replace->insert(Rep);

      if (const Stmt* Else = IfS->getElse()) {
        Replacement Rep(*(Result.SourceManager), Else->getLocStart(), 0,
                        "// the 'else' part\n");
        Replace->insert(Rep);
      }
    }
  }

 private:
  Replacements* Replace;
};

int main(int argc, const char** argv) {
  clang::CommonOptionsParser op(argc, argv, clang::ToolingSampleCategory);
  clang::RefactoringTool Tool(op.getCompilations(), op.getSourcePathList());

  // Set up AST matcher callbacks.
  IfStmtHandler HandlerForIf(&Tool.getReplacements());

  MatchFinder Finder;
  Finder.addMatcher(ifStmt().bind("ifStmt"), &HandlerForIf);

  // Run the tool and collect a list of replacements. We could call
  // runAndSave, which would destructively overwrite the files with
  // their new contents. However, for demonstration purposes it's
  // interesting to show the replacements.
  if (int Result = Tool.run(newFrontendActionFactory(&Finder).get())) {
    return Result;
  }

  llvm::outs() << "Replacements collected by the tool:\n";
  for (auto& r : Tool.getReplacements()) {
    llvm::outs() << r.toString() << "\n";
  }

  return 0;
}
