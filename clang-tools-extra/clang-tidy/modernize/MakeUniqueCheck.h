//===--- MakeUniqueCheck.h - clang-tidy--------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_UNIQUE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_UNIQUE_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace modernize {

/// Replace the pattern:
/// \code
///   std::unique_ptr<type>(new type(args...))
/// \endcode
///
/// With the C++14 version:
/// \code
///   std::make_unique<type>(args...)
/// \endcode
class MakeUniqueCheck : public ClangTidyCheck {
public:
  MakeUniqueCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_UNIQUE_H

