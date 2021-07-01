// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MIGRATE_CPP_CPP_REFACTORING_MATCHER_H_
#define MIGRATE_CPP_CPP_REFACTORING_MATCHER_H_

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Core/Replacement.h"

namespace Carbon {

// This is an abstract class with helpers to make it easier to write matchers.
class Matcher {
 public:
  using ReplacementMap = std::map<std::string, clang::tooling::Replacements>;

  Matcher(const clang::ast_matchers::MatchFinder::MatchResult& in_match_result,
          ReplacementMap* in_replacements)
      : match_result(in_match_result), replacements(in_replacements) {}
  virtual ~Matcher() {}

  // Children must implement this for the main execution.
  virtual void Run() = 0;

 protected:
  // Replaces the given range with the specified text.
  void AddReplacement(clang::CharSourceRange range,
                      llvm::StringRef replacement_text);

  const clang::SourceManager& GetSource() {
    return *match_result.SourceManager;
  }

  template <typename NodeType>
  auto GetNodeOrDie(llvm::StringRef id) -> const NodeType& {
    auto* node = match_result.Nodes.getNodeAs<NodeType>(id);
    if (!node) {
      llvm::report_fatal_error(std::string("getNodeAs failed for ") + id);
    }
    return *node;
  }

 private:
  const clang::ast_matchers::MatchFinder::MatchResult& match_result;
  ReplacementMap* const replacements;
};

}  // namespace Carbon

#endif  // MIGRATE_CPP_CPP_REFACTORING_MATCHER_H_
