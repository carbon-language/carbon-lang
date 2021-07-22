// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/cpp_refactoring/fn_inserter.h"

#include "clang/ASTMatchers/ASTMatchers.h"

namespace cam = ::clang::ast_matchers;

namespace Carbon {

static constexpr char Label[] = "FnInserter";

void FnInserter::Run() {
  const auto& decl = GetNodeAsOrDie<clang::FunctionDecl>(Label);

  // For names like "Class::Method", replace up to "Class" not "Method".
  clang::NestedNameSpecifierLoc qual_loc = decl.getQualifierLoc();
  clang::SourceLocation name_begin_loc =
      qual_loc.hasQualifier() ? qual_loc.getBeginLoc() : decl.getLocation();
  auto range =
      clang::CharSourceRange::getCharRange(decl.getBeginLoc(), name_begin_loc);

  // In order to handle keywords like "virtual" in "virtual auto Foo() -> ...",
  // scan the replaced text and only drop auto/void entries.
  llvm::SmallVector<llvm::StringRef> split;
  GetSourceText(range).split(split, ' ', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  std::string new_text = "fn ";
  for (llvm::StringRef t : split) {
    if (t != "auto" && t != "void") {
      new_text += t.str() + " ";
    }
  }
  AddReplacement(range, new_text);
}

void FnInserterFactory::AddMatcher(cam::MatchFinder* finder,
                                   cam::MatchFinder::MatchCallback* callback) {
  finder->addMatcher(
      cam::functionDecl(cam::anyOf(cam::hasTrailingReturn(),
                                   cam::returns(cam::asString("void"))),
                        cam::unless(cam::anyOf(cam::cxxConstructorDecl(),
                                               cam::cxxDestructorDecl())))
          .bind(Label),
      callback);
}

}  // namespace Carbon
