//===-- AnalysisManager.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"

using namespace clang;
using namespace ento;

void AnalysisManager::anchor() { }

AnalysisManager::AnalysisManager(ASTContext &ctx, DiagnosticsEngine &diags,
                                 const LangOptions &lang,
                                 PathDiagnosticConsumer *pd,
                                 StoreManagerCreator storemgr,
                                 ConstraintManagerCreator constraintmgr, 
                                 CheckerManager *checkerMgr,
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
                                 bool NoRetry)
  : AnaCtxMgr(useUnoptimizedCFG, addImplicitDtors, addInitializers),
    Ctx(ctx), Diags(diags), LangOpts(lang), PD(pd),
    CreateStoreMgr(storemgr), CreateConstraintMgr(constraintmgr),
    CheckerMgr(checkerMgr), 
    MaxNodes(maxnodes), MaxVisit(maxvisit),
    VisualizeEGDot(vizdot), VisualizeEGUbi(vizubi), PurgeDead(purge),
    EagerlyAssume(eager), TrimGraph(trim),
    EagerlyTrimEGraph(eagerlyTrimEGraph),
    IPAMode(ipa),
    InlineMaxStackDepth(inlineMaxStack),
    InlineMaxFunctionSize(inlineMaxFunctionSize),
    InliningMode(IMode),
    NoRetryExhausted(NoRetry)
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
    NoRetryExhausted(ParentAM.NoRetryExhausted)
{
  AnaCtxMgr.getCFGBuildOptions().setAllAlwaysAdd();
}
