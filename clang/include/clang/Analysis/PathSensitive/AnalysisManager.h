//== AnalysisManager.h - Path sensitive analysis data manager ------*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the AnalysisManager class that manages the data and policy
// for path sensitive analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_ANALYSISMANAGER_H
#define LLVM_CLANG_ANALYSIS_ANALYSISMANAGER_H

#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/PathSensitive/AnalysisContext.h"
#include "clang/Analysis/PathDiagnostic.h"

namespace clang {

class AnalysisManager : public BugReporterData {
  AnalysisContextManager AnaCtxMgr;
  LocationContextManager LocCtxMgr;

  ASTContext &Ctx;
  Diagnostic &Diags;
  const LangOptions &LangInfo;

  llvm::OwningPtr<PathDiagnosticClient> PD;

  // Configurable components creators.
  StoreManagerCreator CreateStoreMgr;
  ConstraintManagerCreator CreateConstraintMgr;

  enum AnalysisScope { ScopeTU, ScopeDecl } AScope;

  bool VisualizeEGDot;
  bool VisualizeEGUbi;
  bool PurgeDead;

  /// EargerlyAssume - A flag indicating how the engine should handle
  //   expressions such as: 'x = (y != 0)'.  When this flag is true then
  //   the subexpression 'y != 0' will be eagerly assumed to be true or false,
  //   thus evaluating it to the integers 0 or 1 respectively.  The upside
  //   is that this can increase analysis precision until we have a better way
  //   to lazily evaluate such logic.  The downside is that it eagerly
  //   bifurcates paths.
  bool EagerlyAssume;
  bool TrimGraph;

public:
  AnalysisManager(ASTContext &ctx, Diagnostic &diags, 
                  const LangOptions &lang, PathDiagnosticClient *pd,
                  StoreManagerCreator storemgr,
                  ConstraintManagerCreator constraintmgr,
                  bool vizdot, bool vizubi, bool purge, bool eager, bool trim)

    : Ctx(ctx), Diags(diags), LangInfo(lang), PD(pd),
      CreateStoreMgr(storemgr), CreateConstraintMgr(constraintmgr),
      AScope(ScopeDecl),
      VisualizeEGDot(vizdot), VisualizeEGUbi(vizubi), PurgeDead(purge),
      EagerlyAssume(eager), TrimGraph(trim) {}
  
  ~AnalysisManager() { FlushDiagnostics(); }
  
  void ClearContexts() {
    LocCtxMgr.clear();
    AnaCtxMgr.clear();
  }

  StoreManagerCreator getStoreManagerCreator() {
    return CreateStoreMgr;
  }

  ConstraintManagerCreator getConstraintManagerCreator() {
    return CreateConstraintMgr;
  }

  virtual ASTContext &getASTContext() {
    return Ctx;
  }

  virtual SourceManager &getSourceManager() {
    return getASTContext().getSourceManager();
  }

  virtual Diagnostic &getDiagnostic() {
    return Diags;
  }

  const LangOptions &getLangOptions() const {
    return LangInfo;
  }

  virtual PathDiagnosticClient *getPathDiagnosticClient() {
    return PD.get();
  }
  
  void FlushDiagnostics() {
    if (PD.get())
      PD->FlushDiagnostics();
  }

  bool shouldVisualizeGraphviz() const { return VisualizeEGDot; }

  bool shouldVisualizeUbigraph() const { return VisualizeEGUbi; }

  bool shouldVisualize() const {
    return VisualizeEGDot || VisualizeEGUbi;
  }

  bool shouldTrimGraph() const { return TrimGraph; }

  bool shouldPurgeDead() const { return PurgeDead; }

  bool shouldEagerlyAssume() const { return EagerlyAssume; }

  CFG *getCFG(Decl const *D) {
    return AnaCtxMgr.getContext(D)->getCFG();
  }

  LiveVariables *getLiveVariables(Decl const *D) {
    return AnaCtxMgr.getContext(D)->getLiveVariables();
  }

  ParentMap &getParentMap(Decl const *D) {
    return AnaCtxMgr.getContext(D)->getParentMap();
  }

  // Get the top level stack frame.
  const StackFrameContext *getStackFrame(Decl const *D) {
    return LocCtxMgr.getStackFrame(AnaCtxMgr.getContext(D), 0, 0, 0, 0);
  }

  // Get a stack frame with parent.
  StackFrameContext const *getStackFrame(Decl const *D, 
                                         LocationContext const *Parent,
                                         Stmt const *S, const CFGBlock *Blk,
                                         unsigned Idx) {
    return LocCtxMgr.getStackFrame(AnaCtxMgr.getContext(D), Parent, S, Blk,Idx);
  }
};

}

#endif
