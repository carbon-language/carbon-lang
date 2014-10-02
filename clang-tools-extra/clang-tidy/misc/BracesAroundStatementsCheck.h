//===--- BracesAroundStatementsCheck.h - clang-tidy -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_BRACES_AROUND_STATEMENTS_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_BRACES_AROUND_STATEMENTS_CHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {

class BracesAroundStatementsCheck : public ClangTidyCheck {
public:
  BracesAroundStatementsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void checkStmt(const ast_matchers::MatchFinder::MatchResult &Result,
                 const Stmt *S, SourceLocation StartLoc,
                 SourceLocation EndLocHint = SourceLocation());
  template <typename IfOrWhileStmt>
  SourceLocation findRParenLoc(const IfOrWhileStmt *S, const SourceManager &SM,
                               const ASTContext *Context);
};

} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_BRACES_AROUND_STATEMENTS_CHECK_H
