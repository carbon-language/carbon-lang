//===--- ContainerSizeEmpty.h - clang-tidy ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONTAINERSIZEEMPTY_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONTAINERSIZEEMPTY_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// \brief Checks whether a call to the \c size() method can be replaced with a
/// call to \c empty().
///
/// The emptiness of a container should be checked using the \c empty() method
/// instead of the \c size() method. It is not guaranteed that \c size() is a
/// constant-time function, and it is generally more efficient and also shows
/// clearer intent to use \c empty(). Furthermore some containers may implement
/// the \c empty() method but not implement the \c size() method. Using \c
/// empty() whenever possible makes it easier to switch to another container in
/// the future.
class ContainerSizeEmptyCheck : public ClangTidyCheck {
public:
  ContainerSizeEmptyCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONTAINERSIZEEMPTY_H
