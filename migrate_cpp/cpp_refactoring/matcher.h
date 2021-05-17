// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MIGRATE_CPP_CPP_REFACTORING_MATCHER_H_
#define MIGRATE_CPP_CPP_REFACTORING_MATCHER_H_

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Core/Replacement.h"

namespace Carbon {

// This is an abstract class with helpers to make it easier to write matchers.
class Matcher : public clang::ast_matchers::MatchFinder::MatchCallback {
 public:
  // Alias these to elide the namespaces in subclass headers.
  using MatchFinder = clang::ast_matchers::MatchFinder;
  using Replacements = clang::tooling::Replacements;

  explicit Matcher(std::map<std::string, Replacements>& in_replacements)
      : replacements(&in_replacements) {}

 protected:
  // Replaces the given range with the specified text.
  void AddReplacement(const clang::SourceManager& sm,
                      clang::CharSourceRange range,
                      llvm::StringRef replacement_text);

 private:
  std::map<std::string, Replacements>* const replacements;
};

}  // namespace Carbon

#endif  // MIGRATE_CPP_CPP_REFACTORING_MATCHER_H_
