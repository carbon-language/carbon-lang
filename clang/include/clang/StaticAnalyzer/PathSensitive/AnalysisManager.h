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

#ifndef LLVM_CLANG_GR_ANALYSISMANAGER_H
#define LLVM_CLANG_GR_ANALYSISMANAGER_H

#include "clang/Analysis/AnalysisContext.h"
#include "clang/StaticAnalyzer/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/BugReporter/PathDiagnostic.h"

namespace clang {

namespace idx { 
  class Indexer;
  class TranslationUnit; 
}

namespace ento {

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

  /// \brief Provide function definitions in other translation units. This is
  /// NULL if we don't have multiple translation units. AnalysisManager does
  /// not own the Indexer.
  idx::Indexer *Idxer;

  enum AnalysisScope { ScopeTU, ScopeDecl } AScope;

  // The maximum number of exploded nodes the analyzer will generate.
  unsigned MaxNodes;

  // The maximum number of times the analyzer visit a block.
  unsigned MaxVisit;

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
  bool InlineCall;

public:
  AnalysisManager(ASTContext &ctx, Diagnostic &diags, 
                  const LangOptions &lang, PathDiagnosticClient *pd,
                  StoreManagerCreator storemgr,
                  ConstraintManagerCreator constraintmgr, 
                  idx::Indexer *idxer,
                  unsigned maxnodes, unsigned maxvisit,
                  bool vizdot, bool vizubi, bool purge, bool eager, bool trim,
                  bool inlinecall, bool useUnoptimizedCFG,
                  bool addImplicitDtors, bool addInitializers)

    : AnaCtxMgr(useUnoptimizedCFG, addImplicitDtors, addInitializers),
      Ctx(ctx), Diags(diags), LangInfo(lang), PD(pd),
      CreateStoreMgr(storemgr), CreateConstraintMgr(constraintmgr),Idxer(idxer),
      AScope(ScopeDecl), MaxNodes(maxnodes), MaxVisit(maxvisit),
      VisualizeEGDot(vizdot), VisualizeEGUbi(vizubi), PurgeDead(purge),
      EagerlyAssume(eager), TrimGraph(trim), InlineCall(inlinecall) {}
  
  ~AnalysisManager() { FlushDiagnostics(); }
  
  void ClearContexts() {
    LocCtxMgr.clear();
    AnaCtxMgr.clear();
  }
  
  AnalysisContextManager& getAnalysisContextManager() {
    return AnaCtxMgr;
  }

  StoreManagerCreator getStoreManagerCreator() {
    return CreateStoreMgr;
  }

  ConstraintManagerCreator getConstraintManagerCreator() {
    return CreateConstraintMgr;
  }

  idx::Indexer *getIndexer() const { return Idxer; }

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

  unsigned getMaxNodes() const { return MaxNodes; }

  unsigned getMaxVisit() const { return MaxVisit; }

  bool shouldVisualizeGraphviz() const { return VisualizeEGDot; }

  bool shouldVisualizeUbigraph() const { return VisualizeEGUbi; }

  bool shouldVisualize() const {
    return VisualizeEGDot || VisualizeEGUbi;
  }

  bool shouldTrimGraph() const { return TrimGraph; }

  bool shouldPurgeDead() const { return PurgeDead; }

  bool shouldEagerlyAssume() const { return EagerlyAssume; }

  bool shouldInlineCall() const { return InlineCall; }

  bool hasIndexer() const { return Idxer != 0; }

  AnalysisContext *getAnalysisContextInAnotherTU(const Decl *D);

  CFG *getCFG(Decl const *D) {
    return AnaCtxMgr.getContext(D)->getCFG();
  }

  LiveVariables *getLiveVariables(Decl const *D) {
    return AnaCtxMgr.getContext(D)->getLiveVariables();
  }

  ParentMap &getParentMap(Decl const *D) {
    return AnaCtxMgr.getContext(D)->getParentMap();
  }

  AnalysisContext *getAnalysisContext(const Decl *D) {
    return AnaCtxMgr.getContext(D);
  }

  AnalysisContext *getAnalysisContext(const Decl *D, idx::TranslationUnit *TU) {
    return AnaCtxMgr.getContext(D, TU);
  }

  const StackFrameContext *getStackFrame(AnalysisContext *Ctx,
                                         LocationContext const *Parent,
                                         const Stmt *S,
                                         const CFGBlock *Blk, unsigned Idx) {
    return LocCtxMgr.getStackFrame(Ctx, Parent, S, Blk, Idx);
  }

  // Get the top level stack frame.
  const StackFrameContext *getStackFrame(Decl const *D, 
                                         idx::TranslationUnit *TU) {
    return LocCtxMgr.getStackFrame(AnaCtxMgr.getContext(D, TU), 0, 0, 0, 0);
  }

  // Get a stack frame with parent.
  StackFrameContext const *getStackFrame(const Decl *D, 
                                         LocationContext const *Parent,
                                         const Stmt *S,
                                         const CFGBlock *Blk, unsigned Idx) {
    return LocCtxMgr.getStackFrame(AnaCtxMgr.getContext(D), Parent, S,
                                   Blk,Idx);
  }
};

} // end GR namespace

} // end clang namespace

#endif
