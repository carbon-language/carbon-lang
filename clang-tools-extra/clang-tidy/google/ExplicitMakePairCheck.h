//===--- ExplicitMakePairCheck.h - clang-tidy -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_EXPLICITMAKEPAIRCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_EXPLICITMAKEPAIRCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace google {
namespace build {

/// Check that `make_pair`'s template arguments are deduced.
///
/// G++ 4.6 in C++11 mode fails badly if `make_pair`'s template arguments are
/// specified explicitly, and such use isn't intended in any case.
///
/// Corresponding cpplint.py check name: 'build/explicit_make_pair'.
class ExplicitMakePairCheck : public ClangTidyCheck {
public:
  ExplicitMakePairCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace build
} // namespace google
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_EXPLICITMAKEPAIRCHECK_H
