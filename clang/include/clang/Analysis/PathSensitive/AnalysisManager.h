//== AnalysisManager.cpp - Path sensitive analysis data manager ----*- C++ -*-//
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
  AnalysisContext *CurrentContext;

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

    CurrentContext = ContextMgr.getContext(d);
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

    CurrentContext = 0;
  }
    
  Decl *getCodeDecl() const { 
    assert (AScope == ScopeDecl);
    return CurrentContext->getDecl();
  }
    
  Stmt *getBody() const {
    assert (AScope == ScopeDecl);
    return CurrentContext->getBody();
  }
    
  StoreManagerCreator getStoreManagerCreator() {
    return CreateStoreMgr;
  };

  ConstraintManagerCreator getConstraintManagerCreator() {
    return CreateConstraintMgr;
  }
    
  virtual CFG *getCFG() {
    return CurrentContext->getCFG();
  }
    
  virtual ParentMap &getParentMap() {
    return CurrentContext->getParentMap();
  }

  virtual LiveVariables *getLiveVariables() {
    return CurrentContext->getLiveVariables();
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
