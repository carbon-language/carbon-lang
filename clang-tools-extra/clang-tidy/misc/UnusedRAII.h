//===--- UnusedRAII.h - clang-tidy ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNUSED_RAII_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNUSED_RAII_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {

/// \brief Finds temporaries that look like RAII objects.
///
/// The canonical example for this is a scoped lock.
/// \code
///   {
///     scoped_lock(&global_mutex);
///     critical_section();
///   }
/// \endcode
/// The destructor of the scoped_lock is called before the critical_section is
/// entered, leaving it unprotected.
///
/// We apply a number of heuristics to reduce the false positive count of this
/// check:
///   - Ignore code expanded from macros. Testing frameworks make heavy use of
///     this.
///   - Ignore types with no user-declared constructor. Those are very unlikely
///     to be RAII objects.
///   - Ignore objects at the end of a compound statement (doesn't change behavior).
///   - Ignore objects returned from a call.
class UnusedRAIICheck : public ClangTidyCheck {
public:
  UnusedRAIICheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNUSED_RAII_H
