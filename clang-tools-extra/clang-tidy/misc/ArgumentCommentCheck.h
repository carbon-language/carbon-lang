//===--- ArgumentCommentCheck.h - clang-tidy --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_ARG_COMMENT_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_ARG_COMMENT_CHECK_H

#include "../ClangTidy.h"
#include "llvm/Support/Regex.h"

namespace clang {
namespace tidy {

/// \brief Checks that argument comments match parameter names.
class ArgumentCommentCheck : public ClangTidyCheck {
public:
  ArgumentCommentCheck(StringRef Name, ClangTidyContext *Context);

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  llvm::Regex IdentRE;

  bool isLikelyTypo(llvm::ArrayRef<ParmVarDecl *> Params, StringRef ArgName,
                    unsigned ArgIndex);
  std::vector<std::pair<SourceLocation, StringRef>>
  getCommentsInRange(ASTContext *Ctx, SourceRange Range);
  void checkCallArgs(ASTContext *Ctx, const FunctionDecl *Callee,
                     SourceLocation ArgBeginLoc,
                     llvm::ArrayRef<const Expr *> Args);
};

} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_ARG_COMMENT_CHECK_H
