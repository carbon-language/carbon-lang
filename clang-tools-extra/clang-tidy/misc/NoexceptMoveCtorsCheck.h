//===--- NoexceptMoveCtorsCheck.h - clang-tidy-------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NOEXCEPT_MOVE_CTORS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NOEXCEPT_MOVE_CTORS_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {

/// \brief The check flags move constructors and assignment operators not marked
/// with \c noexcept or marked with \c noexcept(expr) where \c expr evaluates to
/// \c false (but is not a \c false literal itself).
///
/// Move constructors of all the types used with STL containers, for example,
/// need to be declared \c noexcept. Otherwise STL will choose copy constructors
/// instead. The same is valid for move assignment operations.
class NoexceptMoveCtorsCheck : public ClangTidyCheck {
public:
  NoexceptMoveCtorsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NOEXCEPT_MOVE_CTORS_H

