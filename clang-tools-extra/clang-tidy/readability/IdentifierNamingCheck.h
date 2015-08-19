//===--- IdentifierNamingCheck.h - clang-tidy -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IDENTIFIERNAMINGCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IDENTIFIERNAMINGCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// \brief Checks for identifiers naming style mismatch.
class IdentifierNamingCheck : public ClangTidyCheck {
public:
  IdentifierNamingCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;

  enum CaseType {
    CT_AnyCase = 0,
    CT_LowerCase,
    CT_CamelBack,
    CT_UpperCase,
    CT_CamelCase,
  };

  struct NamingStyle {
    NamingStyle() : Case(CT_AnyCase) {}

    NamingStyle(CaseType Case, const std::string &Prefix,
                const std::string &Suffix)
        : Case(Case), Prefix(Prefix), Suffix(Suffix) {}

    CaseType Case;
    std::string Prefix;
    std::string Suffix;

    bool isSet() const {
      return !(Case == CT_AnyCase && Prefix.empty() && Suffix.empty());
    }
  };

private:
  std::vector<NamingStyle> NamingStyles;
  bool IgnoreFailedSplit;

  struct NamingCheckFailure {
    std::string KindName;
    std::string Fixup;
    bool ShouldFix;
    std::vector<SourceRange> Usages;

    NamingCheckFailure() : ShouldFix(true) {}
  };

  llvm::DenseMap<const NamedDecl *, NamingCheckFailure> NamingCheckFailures;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IDENTIFIERNAMINGCHECK_H
