// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MIGRATE_CPP_CPP_REFACTORING_MATCHER_MANAGER_H_
#define MIGRATE_CPP_CPP_REFACTORING_MATCHER_MANAGER_H_

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Core/Replacement.h"
#include "migrate_cpp/cpp_refactoring/matcher.h"

namespace Carbon {

// Manages registration of AST matchers.
class MatcherManager {
 public:
  explicit MatcherManager(Matcher::ReplacementMap* in_replacements)
      : replacements(in_replacements) {}

  template <typename MatcherType>
  void Register() {
    matchers.push_back(
        std::make_unique<MatchCallbackWrapper<MatcherType>>(replacements));
    finder.addMatcher(MatcherType::GetAstMatcher(), matchers.back().get());
  }

  auto GetFinder() -> clang::ast_matchers::MatchFinder* { return &finder; }

 private:
  // Adapts Matcher for use with MatchCallback.
  template <typename MatcherType>
  class MatchCallbackWrapper
      : public clang::ast_matchers::MatchFinder::MatchCallback {
   public:
    explicit MatchCallbackWrapper(Matcher::ReplacementMap* in_replacements)
        : replacements(in_replacements) {}

    void run(const clang::ast_matchers::MatchFinder::MatchResult& match_result)
        override {
      MatcherType matcher(match_result, replacements);
      matcher.Run();
    }

    Matcher::ReplacementMap* const replacements;
  };

  Matcher::ReplacementMap* const replacements;
  clang::ast_matchers::MatchFinder finder;
  std::vector<std::unique_ptr<clang::ast_matchers::MatchFinder::MatchCallback>>
      matchers;
};

}  // namespace Carbon

#endif  // MIGRATE_CPP_CPP_REFACTORING_MATCHER_MANAGER_H_
