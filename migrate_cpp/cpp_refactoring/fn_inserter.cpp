// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/cpp_refactoring/fn_inserter.h"

#include "clang/ASTMatchers/ASTMatchers.h"

namespace cam = ::clang::ast_matchers;

namespace Carbon {

FnInserter::FnInserter(std::map<std::string, Replacements>& in_replacements,
                       cam::MatchFinder* finder)
    : Matcher(in_replacements) {
  // TODO: Switch from isExpansionInMainFile to isDefinition. That should then
  // include `for (const auto* redecl : func->redecls())` to generate
  // replacements.
  finder->addMatcher(
      cam::functionDecl(cam::isDefinition(), cam::hasTrailingReturn())
          .bind(Label),
      this);
}

void FnInserter::Process(const clang::SourceManager& sm,
                         const clang::FunctionDecl& func) {
  auto begin = func.getBeginLoc();
  // Replace the first token in the range, `auto`.
  auto range = clang::CharSourceRange::getTokenRange(begin, begin);
  AddReplacement(sm, range, "fn");
}

void FnInserter::run(const cam::MatchFinder::MatchResult& result) {
  const auto* func = result.Nodes.getNodeAs<clang::FunctionDecl>(Label);
  if (!func) {
    llvm::report_fatal_error(std::string("getNodeAs failed for ") + Label);
  }
  const auto& sm = *(result.SourceManager);
  Process(sm, *func);
  for (const auto* redecl : func->redecls()) {
    Process(sm, *redecl);
  }
}

}  // namespace Carbon
