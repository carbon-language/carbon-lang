//===--- MemsetZeroLengthCheck.h - clang-tidy ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_MEMSET_ZERO_LENGTH_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_MEMSET_ZERO_LENGTH_CHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace runtime {

/// \brief Finds calls to memset with a literal zero in the length argument.
///
/// This is most likely unintended and the length and value arguments are
/// swapped.
///
/// Corresponding cpplint.py check name: 'runtime/memset'.
class MemsetZeroLengthCheck : public ClangTidyCheck {
public:
  MemsetZeroLengthCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace runtime
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_MEMSET_ZERO_LENGTH_CHECK_H
