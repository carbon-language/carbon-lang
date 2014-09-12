//===--- UsingNamespaceDirectiveCheck.h - clang-tidy ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_USING_NAMESPACE_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_USING_NAMESPACE_CHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace build {

/// \brief Finds using namespace directives.
///
/// http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml?showone=Namespaces#Namespaces
/// Corresponding cpplint.py check name: 'build/namespaces'.
class UsingNamespaceDirectiveCheck : public ClangTidyCheck {
public:
  UsingNamespaceDirectiveCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace build
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_USING_NAMESPACE_CHECK_H
