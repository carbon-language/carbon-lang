//===---------- NamespaceAliaser.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NamespaceAliaser.h"

#include "ASTUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"
namespace clang {
namespace tidy {
namespace utils {

using namespace ast_matchers;

NamespaceAliaser::NamespaceAliaser(const SourceManager &SourceMgr)
    : SourceMgr(SourceMgr) {}

AST_MATCHER_P(NamespaceAliasDecl, hasTargetNamespace,
              ast_matchers::internal::Matcher<NamespaceDecl>, innerMatcher) {
  return innerMatcher.matches(*Node.getNamespace(), Finder, Builder);
}

Optional<FixItHint>
NamespaceAliaser::createAlias(ASTContext &Context, const Stmt &Statement,
                              StringRef Namespace,
                              const std::vector<std::string> &Abbreviations) {
  const FunctionDecl *Function = getSurroundingFunction(Context, Statement);
  if (!Function || !Function->hasBody())
    return None;

  if (AddedAliases[Function].count(Namespace.str()) != 0)
    return None;

  // FIXME: Doesn't consider the order of declarations.
  // If we accidentally pick an alias defined later in the function,
  // the output won't compile.
  // FIXME: Also doesn't consider file or class-scope aliases.

  const auto *ExistingAlias = selectFirst<NamedDecl>(
      "alias", match(functionDecl(hasBody(compoundStmt(has(declStmt(
                         has(namespaceAliasDecl(hasTargetNamespace(hasName(
                                                    std::string(Namespace))))
                                 .bind("alias"))))))),
                     *Function, Context));

  if (ExistingAlias != nullptr) {
    AddedAliases[Function][Namespace.str()] = ExistingAlias->getName().str();
    return None;
  }

  for (const auto &Abbreviation : Abbreviations) {
    DeclarationMatcher ConflictMatcher = namedDecl(hasName(Abbreviation));
    const auto HasConflictingChildren =
        !match(findAll(ConflictMatcher), *Function, Context).empty();
    const auto HasConflictingAncestors =
        !match(functionDecl(hasAncestor(decl(has(ConflictMatcher)))), *Function,
               Context)
             .empty();
    if (HasConflictingAncestors || HasConflictingChildren)
      continue;

    std::string Declaration =
        (llvm::Twine("\nnamespace ") + Abbreviation + " = " + Namespace + ";")
            .str();
    SourceLocation Loc =
        Lexer::getLocForEndOfToken(Function->getBody()->getBeginLoc(), 0,
                                   SourceMgr, Context.getLangOpts());
    AddedAliases[Function][Namespace.str()] = Abbreviation;
    return FixItHint::CreateInsertion(Loc, Declaration);
  }

  return None;
}

std::string NamespaceAliaser::getNamespaceName(ASTContext &Context,
                                               const Stmt &Statement,
                                               StringRef Namespace) const {
  const auto *Function = getSurroundingFunction(Context, Statement);
  auto FunctionAliases = AddedAliases.find(Function);
  if (FunctionAliases != AddedAliases.end()) {
    if (FunctionAliases->second.count(Namespace) != 0) {
      return FunctionAliases->second.find(Namespace)->getValue();
    }
  }
  return Namespace.str();
}

} // namespace utils
} // namespace tidy
} // namespace clang
