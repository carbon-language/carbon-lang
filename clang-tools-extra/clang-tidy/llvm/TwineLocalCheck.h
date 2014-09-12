//===--- TwineLocalCheck.h - clang-tidy -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_TWINE_LOCAL_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_TWINE_LOCAL_CHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {

/// \brief Looks for local Twine variables which are prone to use after frees
/// and should be generally avoided.
class TwineLocalCheck : public ClangTidyCheck {
public:
  TwineLocalCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_TWINE_LOCAL_CHECK_H
