//===--- StringReferenceMemberCheck.h - clang-tidy ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_STRING_REF_MEMBER_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_STRING_REF_MEMBER_CHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace runtime {

/// \brief Finds members of type 'const string&'.
///
/// const string reference members are generally considered unsafe as they can
/// be created from a temporary quite easily.
///
/// \code
/// struct S {
///  S(const string &Str) : Str(Str) {}
///  const string &Str;
/// };
/// S instance("string");
/// \endcode
///
/// In the constructor call a string temporary is created from const char * and
/// destroyed immediately after the call. This leaves around a dangling
/// reference.
///
/// This check emit warnings for both std::string and ::string const reference
/// members.
///
/// Corresponding cpplint.py check name: 'runtime/member_string_reference'.
class StringReferenceMemberCheck : public ClangTidyCheck {
public:
  StringReferenceMemberCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace runtime
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_STRING_REF_MEMBER_CHECK_H
