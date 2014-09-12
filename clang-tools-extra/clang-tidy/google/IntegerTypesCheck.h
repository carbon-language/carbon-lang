//===--- IntegerTypesCheck.h - clang-tidy -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_INTEGERTYPESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_INTEGERTYPESCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace runtime {

/// \brief Finds uses of short, long and long long and suggest replacing them
/// with u?intXX(_t)?.
/// Correspondig cpplint.py check: runtime/int.
class IntegerTypesCheck : public ClangTidyCheck {
public:
  IntegerTypesCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context), UnsignedTypePrefix("uint"),
        SignedTypePrefix("int"), AddUnderscoreT(false) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const StringRef UnsignedTypePrefix;
  const StringRef SignedTypePrefix;
  const bool AddUnderscoreT;
};

} // namespace runtime
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_INTEGERTYPESCHECK_H
