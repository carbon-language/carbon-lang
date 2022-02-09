//===--- UnnamedNamespaceInHeaderCheck.cpp - clang-tidy ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnnamedNamespaceInHeaderCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace google {
namespace build {

UnnamedNamespaceInHeaderCheck::UnnamedNamespaceInHeaderCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      RawStringHeaderFileExtensions(Options.getLocalOrGlobal(
          "HeaderFileExtensions", utils::defaultHeaderFileExtensions())) {
  if (!utils::parseFileExtensions(RawStringHeaderFileExtensions,
                                  HeaderFileExtensions,
                                  utils::defaultFileExtensionDelimiters())) {
    this->configurationDiag("Invalid header file extension: '%0'")
        << RawStringHeaderFileExtensions;
  }
}

void UnnamedNamespaceInHeaderCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "HeaderFileExtensions", RawStringHeaderFileExtensions);
}

void UnnamedNamespaceInHeaderCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
    Finder->addMatcher(namespaceDecl(isAnonymous()).bind("anonymousNamespace"),
                       this);
}

void UnnamedNamespaceInHeaderCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *N = Result.Nodes.getNodeAs<NamespaceDecl>("anonymousNamespace");
  SourceLocation Loc = N->getBeginLoc();
  if (!Loc.isValid())
    return;

  if (utils::isPresumedLocInHeaderFile(Loc, *Result.SourceManager,
                                       HeaderFileExtensions))
    diag(Loc, "do not use unnamed namespaces in header files");
}

} // namespace build
} // namespace google
} // namespace tidy
} // namespace clang
