//===--- ExplicitConstructorCheck.h - clang-tidy ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_EXPLICIT_CONSTRUCTOR_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_EXPLICIT_CONSTRUCTOR_CHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {

/// \brief Checks that all single-argument constructors are explicit.
///
/// see:
/// http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml#Explicit_Constructors
class ExplicitConstructorCheck : public ClangTidyCheck {
public:
  ExplicitConstructorCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_EXPLICIT_CONSTRUCTOR_CHECK_H
