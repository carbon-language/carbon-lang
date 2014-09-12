//===--- NamedParameterCheck.h - clang-tidy ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_NAMED_PARAMETER_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_NAMED_PARAMETER_CHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// \brief Find functions with unnamed arguments.
///
/// http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml?showone=Function_Declarations_and_Definitions#Function_Declarations_and_Definitions
/// Corresponding cpplint.py check name: 'readability/function'.
class NamedParameterCheck : public ClangTidyCheck {
public:
  NamedParameterCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_NAMED_PARAMETER_CHECK_H
