// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/cpp_refactoring/var_decl.h"

#include "clang/ASTMatchers/ASTMatchers.h"

namespace cam = ::clang::ast_matchers;

namespace Carbon {

VarDecl::VarDecl(std::map<std::string, Replacements>& in_replacements,
                 cam::MatchFinder* finder)
    : Matcher(in_replacements) {
  finder->addMatcher(cam::varDecl().bind(Label), this);
}

void VarDecl::run(const cam::MatchFinder::MatchResult& result) {
  const auto* decl = result.Nodes.getNodeAs<clang::VarDecl>(Label);
  if (!decl) {
    llvm::report_fatal_error(std::string("getNodeAs failed for ") + Label);
  }
  // Replace the full declaration.
  auto range = clang::CharSourceRange::getTokenRange(decl->getBeginLoc(),
                                                     decl->getEndLoc());
  std::string repl;
  if (result.Nodes.getNodeAs<clang::ParmVarDecl>(Label) == nullptr) {
    // Not a param, so add `var`.
    repl = "var ";
  }
  repl += decl->getNameAsString() + ": " + decl->getType().getAsString();
  AddReplacement(*(result.SourceManager), range, repl);
}

}  // namespace Carbon
