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
                                 const PathDiagnosticConsumers &PDC,
                                 StoreManagerCreator storemgr,
                                 ConstraintManagerCreator constraintmgr, 
                                 CheckerManager *checkerMgr,
                                 const ConfigTable &Config,
                                 unsigned maxnodes, unsigned maxvisit,
                                 bool vizdot, bool vizubi,
                                 AnalysisPurgeMode purge,
                                 bool eager, bool trim,
                                 bool useUnoptimizedCFG,
                                 bool addImplicitDtors,
                                 bool eagerlyTrimEGraph,
                                 AnalysisIPAMode ipa,
                                 unsigned inlineMaxStack,
                                 unsigned inlineMaxFunctionSize,
                                 AnalysisInliningMode IMode,
                                 bool NoRetry)
  : AnaCtxMgr(useUnoptimizedCFG, addImplicitDtors, /*addInitializers=*/true),
    Ctx(ctx), Diags(diags), LangOpts(lang),
    PathConsumers(PDC),
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
    NoRetryExhausted(NoRetry),
    Config(Config)
{
  AnaCtxMgr.getCFGBuildOptions().setAllAlwaysAdd();
}

AnalysisManager::~AnalysisManager() {
  FlushDiagnostics();
  for (PathDiagnosticConsumers::iterator I = PathConsumers.begin(),
       E = PathConsumers.end(); I != E; ++I) {
    delete *I;
  }
}

void AnalysisManager::FlushDiagnostics() {
  PathDiagnosticConsumer::FilesMade filesMade;
  for (PathDiagnosticConsumers::iterator I = PathConsumers.begin(),
       E = PathConsumers.end();
       I != E; ++I) {
    (*I)->FlushDiagnostics(&filesMade);
  }
}
