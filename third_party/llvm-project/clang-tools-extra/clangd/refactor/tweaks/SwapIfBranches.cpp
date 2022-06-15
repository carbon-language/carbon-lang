//===--- SwapIfBranches.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ParsedAST.h"
#include "SourceCode.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace clangd {
namespace {
/// Swaps the 'then' and 'else' branch of the if statement.
/// Before:
///   if (foo) { return 10; } else { continue; }
///   ^^^^^^^                 ^^^^
/// After:
///   if (foo) { continue; } else { return 10; }
class SwapIfBranches : public Tweak {
public:
  const char *id() const override final;

  bool prepare(const Selection &Inputs) override;
  Expected<Effect> apply(const Selection &Inputs) override;
  std::string title() const override { return "Swap if branches"; }
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }
  bool hidden() const override { return true; }

private:
  const IfStmt *If = nullptr;
};

REGISTER_TWEAK(SwapIfBranches)

bool SwapIfBranches::prepare(const Selection &Inputs) {
  for (const SelectionTree::Node *N = Inputs.ASTSelection.commonAncestor();
       N && !If; N = N->Parent) {
    // Stop once we hit a block, e.g. a lambda in the if condition.
    if (llvm::isa_and_nonnull<CompoundStmt>(N->ASTNode.get<Stmt>()))
      return false;
    If = dyn_cast_or_null<IfStmt>(N->ASTNode.get<Stmt>());
  }
  // avoid dealing with single-statement brances, they require careful handling
  // to avoid changing semantics of the code (i.e. dangling else).
  return If && isa_and_nonnull<CompoundStmt>(If->getThen()) &&
         isa_and_nonnull<CompoundStmt>(If->getElse());
}

Expected<Tweak::Effect> SwapIfBranches::apply(const Selection &Inputs) {
  auto &Ctx = Inputs.AST->getASTContext();
  auto &SrcMgr = Inputs.AST->getSourceManager();

  auto ThenRng = toHalfOpenFileRange(SrcMgr, Ctx.getLangOpts(),
                                     If->getThen()->getSourceRange());
  if (!ThenRng)
    return error("Could not obtain range of the 'then' branch. Macros?");
  auto ElseRng = toHalfOpenFileRange(SrcMgr, Ctx.getLangOpts(),
                                     If->getElse()->getSourceRange());
  if (!ElseRng)
    return error("Could not obtain range of the 'else' branch. Macros?");

  auto ThenCode = toSourceCode(SrcMgr, *ThenRng);
  auto ElseCode = toSourceCode(SrcMgr, *ElseRng);

  tooling::Replacements Result;
  if (auto Err = Result.add(tooling::Replacement(Ctx.getSourceManager(),
                                                 ThenRng->getBegin(),
                                                 ThenCode.size(), ElseCode)))
    return std::move(Err);
  if (auto Err = Result.add(tooling::Replacement(Ctx.getSourceManager(),
                                                 ElseRng->getBegin(),
                                                 ElseCode.size(), ThenCode)))
    return std::move(Err);
  return Effect::mainFileEdit(SrcMgr, std::move(Result));
}

} // namespace
} // namespace clangd
} // namespace clang
