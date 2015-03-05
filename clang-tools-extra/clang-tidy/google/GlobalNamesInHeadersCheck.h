//===--- GlobalNamesInHeadersCheck.h - clang-tidy ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_GLOBAL_NAMES_IN_HEADERS_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_GLOBAL_NAMES_IN_HEADERS_CHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace google {
namespace readability {

// Flag global namespace pollution in header files.
// Right now it only triggers on using declarations and directives.
class GlobalNamesInHeadersCheck : public ClangTidyCheck {
public:
  GlobalNamesInHeadersCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace readability
} // namespace google
} // namespace tidy
} // namespace clang

#endif  // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_GLOBAL_NAMES_IN_HEADERS_CHECK_H
