//===--- UncheckedOptionalAccessCheck.cpp - clang-tidy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UncheckedOptionalAccessCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/ControlFlowContext.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/Models/UncheckedOptionalAccessModel.h"
#include "clang/Analysis/FlowSensitive/SourceLocationsLattice.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <vector>

namespace clang {
namespace tidy {
namespace bugprone {
using ast_matchers::MatchFinder;
using dataflow::SourceLocationsLattice;
using dataflow::UncheckedOptionalAccessModel;
using llvm::Optional;

static constexpr llvm::StringLiteral FuncID("fun");

static Optional<SourceLocationsLattice>
analyzeFunction(const FunctionDecl &FuncDecl, ASTContext &ASTCtx) {
  using dataflow::ControlFlowContext;
  using dataflow::DataflowAnalysisState;
  using llvm::Expected;

  Expected<ControlFlowContext> Context =
      ControlFlowContext::build(&FuncDecl, FuncDecl.getBody(), &ASTCtx);
  if (!Context)
    return llvm::None;

  dataflow::DataflowAnalysisContext AnalysisContext(
      std::make_unique<dataflow::WatchedLiteralsSolver>());
  dataflow::Environment Env(AnalysisContext, FuncDecl);
  UncheckedOptionalAccessModel Analysis(ASTCtx);
  Expected<std::vector<Optional<DataflowAnalysisState<SourceLocationsLattice>>>>
      BlockToOutputState =
          dataflow::runDataflowAnalysis(*Context, Analysis, Env);
  if (!BlockToOutputState)
    return llvm::None;
  assert(Context->getCFG().getExit().getBlockID() < BlockToOutputState->size());

  const Optional<DataflowAnalysisState<SourceLocationsLattice>>
      &ExitBlockState =
          (*BlockToOutputState)[Context->getCFG().getExit().getBlockID()];
  // `runDataflowAnalysis` doesn't guarantee that the exit block is visited;
  // for example, when it is unreachable.
  // FIXME: Diagnose violations even when the exit block is unreachable.
  if (!ExitBlockState.hasValue())
    return llvm::None;

  return std::move(ExitBlockState->Lattice);
}

void UncheckedOptionalAccessCheck::registerMatchers(MatchFinder *Finder) {
  using namespace ast_matchers;

  auto HasOptionalCallDescendant = hasDescendant(callExpr(callee(cxxMethodDecl(
      ofClass(UncheckedOptionalAccessModel::optionalClassDecl())))));
  Finder->addMatcher(
      decl(anyOf(functionDecl(unless(isExpansionInSystemHeader()),
                              // FIXME: Remove the filter below when lambdas are
                              // well supported by the check.
                              unless(hasDeclContext(cxxRecordDecl(isLambda()))),
                              hasBody(HasOptionalCallDescendant)),
                 cxxConstructorDecl(hasAnyConstructorInitializer(
                     withInitializer(HasOptionalCallDescendant)))))
          .bind(FuncID),
      this);
}

void UncheckedOptionalAccessCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (Result.SourceManager->getDiagnostics().hasUncompilableErrorOccurred())
    return;

  const auto *FuncDecl = Result.Nodes.getNodeAs<FunctionDecl>(FuncID);
  if (FuncDecl->isTemplated())
    return;

  if (Optional<SourceLocationsLattice> Errors =
          analyzeFunction(*FuncDecl, *Result.Context))
    for (const SourceLocation &Loc : Errors->getSourceLocations())
      diag(Loc, "unchecked access to optional value");
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
