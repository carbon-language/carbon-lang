// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MIGRATE_CPP_CPP_REFACTORING_MATCHER_H_
#define MIGRATE_CPP_CPP_REFACTORING_MATCHER_H_

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Core/Replacement.h"

namespace Carbon {

// This is an abstract class with helpers to make it easier to write matchers.
// Note a MatcherFactory (below) is also typically required.
class Matcher {
 public:
  using ReplacementMap = std::map<std::string, clang::tooling::Replacements>;

  Matcher(const clang::ast_matchers::MatchFinder::MatchResult* in_match_result,
          ReplacementMap* in_replacements)
      : match_result(in_match_result), replacements(in_replacements) {}
  virtual ~Matcher() = default;

  // Performs main execution of the matcher when a result is found.
  //
  // Although Run() will be invoked with a known type, it is virtual to allow
  // earlier compilation checks.
  virtual void Run() = 0;

 protected:
  // Replaces the given range with the specified text.
  void AddReplacement(clang::CharSourceRange range,
                      llvm::StringRef replacement_text);

  // Returns a matched node by ID, exiting if not present.
  template <typename NodeType>
  auto GetNodeAsOrDie(llvm::StringRef id) -> const NodeType& {
    auto* node = match_result->Nodes.getNodeAs<NodeType>(id);
    if (!node) {
      llvm::report_fatal_error(std::string("getNodeAs failed for ") + id);
    }
    return *node;
  }

  // Returns the language options.
  auto GetLangOpts() -> const clang::LangOptions& {
    return match_result->Context->getLangOpts();
  }

  // Returns the full source manager.
  auto GetSources() -> const clang::SourceManager& {
    return *match_result->SourceManager;
  }

  // Returns the source text for a given range.
  auto GetSourceText(clang::CharSourceRange range) -> llvm::StringRef {
    return clang::Lexer::getSourceText(range, GetSources(), GetLangOpts());
  }

 private:
  const clang::ast_matchers::MatchFinder::MatchResult* const match_result;
  ReplacementMap* const replacements;
};

class MatcherFactory {
 public:
  virtual ~MatcherFactory() = default;

  virtual std::unique_ptr<Matcher> CreateMatcher(
      const clang::ast_matchers::MatchFinder::MatchResult* match_result,
      Matcher::ReplacementMap* replacements) = 0;

  // Returns the AST matcher which determines when the Matcher is instantiated
  // and run.
  virtual auto GetAstMatcher() -> clang::ast_matchers::DeclarationMatcher = 0;
};

template <typename MatcherType>
class MatcherFactoryBase : public MatcherFactory {
 public:
  std::unique_ptr<Matcher> CreateMatcher(
      const clang::ast_matchers::MatchFinder::MatchResult* match_result,
      Matcher::ReplacementMap* replacements) override {
    return std::make_unique<MatcherType>(match_result, replacements);
  }
};

}  // namespace Carbon

#endif  // MIGRATE_CPP_CPP_REFACTORING_MATCHER_H_
