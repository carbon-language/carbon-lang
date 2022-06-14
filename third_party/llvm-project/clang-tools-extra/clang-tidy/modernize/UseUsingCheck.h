//===--- UseUsingCheck.h - clang-tidy----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_USING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_USING_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace modernize {

/// Check finds typedefs and replaces it with usings.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-using.html
class UseUsingCheck : public ClangTidyCheck {

  const bool IgnoreMacros;
  SourceLocation LastReplacementEnd;
  llvm::DenseMap<const Decl *, SourceRange> LastTagDeclRanges;

  std::string FirstTypedefType;
  std::string FirstTypedefName;

public:
  UseUsingCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_USING_H
