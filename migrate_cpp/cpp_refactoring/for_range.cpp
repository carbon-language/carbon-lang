// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/cpp_refactoring/for_range.h"

#include "clang/ASTMatchers/ASTMatchers.h"

namespace cam = ::clang::ast_matchers;

namespace Carbon {

static constexpr char Label[] = "ForRange";

void ForRange::Run() {
  const auto& stmt = GetNodeAsOrDie<clang::CXXForRangeStmt>(Label);

  // Wrap `in` with spaces so that `for (auto i:items)` has valid results.
  AddReplacement(clang::CharSourceRange::getTokenRange(stmt.getColonLoc(),
                                                       stmt.getColonLoc()),
                 " in ");
}

void ForRangeFactory::AddMatcher(cam::MatchFinder* finder,
                                 cam::MatchFinder::MatchCallback* callback) {
  finder->addMatcher(cam::cxxForRangeStmt().bind(Label), callback);
}

}  // namespace Carbon
