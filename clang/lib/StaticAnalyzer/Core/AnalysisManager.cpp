//===-- AnalysisManager.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/Index/Entity.h"
#include "clang/Index/Indexer.h"

using namespace clang;
using namespace ento;

void AnalysisManager::anchor() { }

AnalysisManager::AnalysisManager(ASTContext &ctx, DiagnosticsEngine &diags,
                                 const LangOptions &lang,
                                 PathDiagnosticConsumer *pd,
                                 StoreManagerCreator storemgr,
                                 ConstraintManagerCreator constraintmgr, 
                                 CheckerManager *checkerMgr,
                                 idx::Indexer *idxer,
                                 unsigned maxnodes, unsigned maxvisit,
                                 bool vizdot, bool vizubi,
                                 AnalysisPurgeMode purge,
                                 bool eager, bool trim,
                                 bool useUnoptimizedCFG,
                                 bool addImplicitDtors, bool addInitializers,
                                 bool eagerlyTrimEGraph,
                                 AnalysisIPAMode ipa,
                                 unsigned inlineMaxStack,
                                 unsigned inlineMaxFunctionSize,
                                 AnalysisInliningMode IMode,
                                 bool retry)
  : AnaCtxMgr(useUnoptimizedCFG, addImplicitDtors, addInitializers),
    Ctx(ctx), Diags(diags), LangOpts(lang), PD(pd),
    CreateStoreMgr(storemgr), CreateConstraintMgr(constraintmgr),
    CheckerMgr(checkerMgr), Idxer(idxer),
    AScope(ScopeDecl), MaxNodes(maxnodes), MaxVisit(maxvisit),
    VisualizeEGDot(vizdot), VisualizeEGUbi(vizubi), PurgeDead(purge),
    EagerlyAssume(eager), TrimGraph(trim),
    EagerlyTrimEGraph(eagerlyTrimEGraph),
    IPAMode(ipa),
    InlineMaxStackDepth(inlineMaxStack),
    InlineMaxFunctionSize(inlineMaxFunctionSize),
    InliningMode(IMode),
    RetryExhausted(retry)
{
  AnaCtxMgr.getCFGBuildOptions().setAllAlwaysAdd();
}

AnalysisManager::AnalysisManager(ASTContext &ctx, DiagnosticsEngine &diags,
                                 AnalysisManager &ParentAM)
  : AnaCtxMgr(ParentAM.AnaCtxMgr.getUseUnoptimizedCFG(),
              ParentAM.AnaCtxMgr.getCFGBuildOptions().AddImplicitDtors,
              ParentAM.AnaCtxMgr.getCFGBuildOptions().AddInitializers),
    Ctx(ctx), Diags(diags),
    LangOpts(ParentAM.LangOpts), PD(ParentAM.getPathDiagnosticConsumer()),
    CreateStoreMgr(ParentAM.CreateStoreMgr),
    CreateConstraintMgr(ParentAM.CreateConstraintMgr),
    CheckerMgr(ParentAM.CheckerMgr),
    Idxer(ParentAM.Idxer),
    AScope(ScopeDecl),
    MaxNodes(ParentAM.MaxNodes),
    MaxVisit(ParentAM.MaxVisit),
    VisualizeEGDot(ParentAM.VisualizeEGDot),
    VisualizeEGUbi(ParentAM.VisualizeEGUbi),
    PurgeDead(ParentAM.PurgeDead),
    EagerlyAssume(ParentAM.EagerlyAssume),
    TrimGraph(ParentAM.TrimGraph),
    EagerlyTrimEGraph(ParentAM.EagerlyTrimEGraph),
    IPAMode(ParentAM.IPAMode),
    InlineMaxStackDepth(ParentAM.InlineMaxStackDepth),
    InlineMaxFunctionSize(ParentAM.InlineMaxFunctionSize),
    InliningMode(ParentAM.InliningMode),
    RetryExhausted(ParentAM.RetryExhausted)
{
  AnaCtxMgr.getCFGBuildOptions().setAllAlwaysAdd();
}


AnalysisDeclContext *
AnalysisManager::getAnalysisDeclContextInAnotherTU(const Decl *D) {
  idx::Entity Ent = idx::Entity::get(const_cast<Decl *>(D), 
                                     Idxer->getProgram());
  FunctionDecl *FuncDef;
  idx::TranslationUnit *TU;
  llvm::tie(FuncDef, TU) = Idxer->getDefinitionFor(Ent);

  if (FuncDef == 0)
    return 0;

  // This AnalysisDeclContext wraps function definition in another translation unit.
  // But it is still owned by the AnalysisManager associated with the current
  // translation unit.
  return AnaCtxMgr.getContext(FuncDef, TU);
}
