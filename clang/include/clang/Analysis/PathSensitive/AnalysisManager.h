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
  AnalysisContextManager ContextMgr;
  AnalysisContext *RootContext;

  LocationContextManager LocCtxMgr;

  ASTContext &Ctx;
  Diagnostic &Diags;
  const LangOptions &LangInfo;

  llvm::OwningPtr<PathDiagnosticClient> PD;

  // Configurable components creators.
  StoreManagerCreator CreateStoreMgr;
  ConstraintManagerCreator CreateConstraintMgr;

  enum AnalysisScope { ScopeTU, ScopeDecl } AScope;
      
  bool DisplayedFunction;
  bool VisualizeEGDot;
  bool VisualizeEGUbi;
  bool PurgeDead;
  bool EagerlyAssume;
  bool TrimGraph;

public:
  AnalysisManager(Decl *d, ASTContext &ctx, Diagnostic &diags, 
                  const LangOptions &lang, PathDiagnosticClient *pd,
                  StoreManagerCreator storemgr,
                  ConstraintManagerCreator constraintmgr,
                  bool displayProgress, bool vizdot, bool vizubi, 
                  bool purge, bool eager, bool trim)
    : Ctx(ctx), Diags(diags), LangInfo(lang), PD(pd), 
      CreateStoreMgr(storemgr), CreateConstraintMgr(constraintmgr),
      AScope(ScopeDecl), DisplayedFunction(!displayProgress),
      VisualizeEGDot(vizdot), VisualizeEGUbi(vizubi), PurgeDead(purge),
      EagerlyAssume(eager), TrimGraph(trim) {

    RootContext = ContextMgr.getContext(d);
  }
    
  AnalysisManager(ASTContext &ctx, Diagnostic &diags, 
                  const LangOptions &lang, PathDiagnosticClient *pd,
                  StoreManagerCreator storemgr,
                  ConstraintManagerCreator constraintmgr,
                  bool displayProgress, bool vizdot, bool vizubi, 
                  bool purge, bool eager, bool trim)

    : Ctx(ctx), Diags(diags), LangInfo(lang), PD(pd), 
      CreateStoreMgr(storemgr), CreateConstraintMgr(constraintmgr),
      AScope(ScopeDecl), DisplayedFunction(!displayProgress),
      VisualizeEGDot(vizdot), VisualizeEGUbi(vizubi), PurgeDead(purge),
      EagerlyAssume(eager), TrimGraph(trim) {

    RootContext = 0;
  }

  void setContext(Decl *D) {
    RootContext = ContextMgr.getContext(D);
    DisplayedFunction = false;
  }
    
  Decl *getCodeDecl() const { 
    assert (AScope == ScopeDecl);
    return RootContext->getDecl();
  }
    
  Stmt *getBody() const {
    assert (AScope == ScopeDecl);
    return RootContext->getBody();
  }
    
  StoreManagerCreator getStoreManagerCreator() {
    return CreateStoreMgr;
  };

  ConstraintManagerCreator getConstraintManagerCreator() {
    return CreateConstraintMgr;
  }
    
  virtual CFG *getCFG() {
    return RootContext->getCFG();
  }
    
  virtual ParentMap &getParentMap() {
    return RootContext->getParentMap();
  }

  virtual LiveVariables *getLiveVariables() {
    return RootContext->getLiveVariables();
  }
    
  virtual ASTContext &getContext() {
    return Ctx;
  }
    
  virtual SourceManager &getSourceManager() {
    return getContext().getSourceManager();
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

  StackFrameContext *getRootStackFrame() {
    return LocCtxMgr.getStackFrame(RootContext, 0, 0);
  }
    
  bool shouldVisualizeGraphviz() const { return VisualizeEGDot; }

  bool shouldVisualizeUbigraph() const { return VisualizeEGUbi; }

  bool shouldVisualize() const {
    return VisualizeEGDot || VisualizeEGUbi;
  }

  bool shouldTrimGraph() const { return TrimGraph; }

  bool shouldPurgeDead() const { return PurgeDead; }

  bool shouldEagerlyAssume() const { return EagerlyAssume; }

  void DisplayFunction();
};

}

#endif
