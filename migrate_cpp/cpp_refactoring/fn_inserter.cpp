// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/cpp_refactoring/fn_inserter.h"

namespace cam = ::clang::ast_matchers;

namespace Carbon {

static constexpr char Label[] = "FnInserter";

cam::DeclarationMatcher FnInserter::GetAstMatcher() {
  return cam::functionDecl(cam::anyOf(cam::hasTrailingReturn(),
                                      cam::returns(cam::asString("void"))),
                           cam::unless(cam::anyOf(cam::cxxConstructorDecl(),
                                                  cam::cxxDestructorDecl())))
      .bind(Label);
}

void FnInserter::Run() {
  const auto& decl = GetNodeOrDie<clang::FunctionDecl>(Label);
  clang::SourceLocation begin = decl.getBeginLoc();
  // Replace the first token in the range, `auto`.
  auto range = clang::CharSourceRange::getTokenRange(begin, begin);
  AddReplacement(range, "fn");
}

}  // namespace Carbon
