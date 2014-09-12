//===--- UseOverride.h - clang-tidy -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_USE_OVERRIDE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_USE_OVERRIDE_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {

/// \brief Use C++11's 'override' and remove 'virtual' where applicable.
class UseOverride : public ClangTidyCheck {
public:
  UseOverride(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_USE_OVERRIDE_H

