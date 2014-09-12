//===--- OverloadedUnaryAndCheck.h - clang-tidy -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_OVERLOADED_UNARY_AND_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_OVERLOADED_UNARY_AND_CHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace runtime {

/// \brief Finds overloads of unary operator &.
///
/// http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml?showone=Operator_Overloading#Operator_Overloading
/// Corresponding cpplint.py check name: 'runtime/operator'.
class OverloadedUnaryAndCheck : public ClangTidyCheck {
public:
  OverloadedUnaryAndCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace runtime
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_OVERLOADED_UNARY_AND_CHECK_H
