//===--- MisplacedConstCheck.h - clang-tidy----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MISPLACED_CONST_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MISPLACED_CONST_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace misc {

/// This check diagnoses when a const qualifier is applied to a typedef to a
/// pointer type rather than to the pointee.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc-misplaced-const.html
class MisplacedConstCheck : public ClangTidyCheck {
public:
  MisplacedConstCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MISPLACED_CONST_H
