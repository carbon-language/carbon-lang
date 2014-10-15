//===--- FunctionSize.h - clang-tidy ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_FUNCTIONSIZE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_FUNCTIONSIZE_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// \brief Checks for large functions based on various metrics.
class FunctionSizeCheck : public ClangTidyCheck {
public:
  FunctionSizeCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;

private:
  struct FunctionInfo {
    FunctionInfo() : Lines(0), Statements(0), Branches(0) {}
    unsigned Lines;
    unsigned Statements;
    unsigned Branches;
  };

  const unsigned LineThreshold;
  const unsigned StatementThreshold;
  const unsigned BranchThreshold;

  llvm::DenseMap<const FunctionDecl *, FunctionInfo> FunctionInfos;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_FUNCTIONSIZE_H
