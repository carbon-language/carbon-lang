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
#include "clang/Frontend/AnalyzerOptions.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"

namespace clang {

namespace idx { 
  class Indexer;
  class TranslationUnit; 
}

namespace ento {
  class CheckerManager;

typedef llvm::SmallPtrSet<const Decl*,24> SetOfDecls;

class FunctionSummariesTy {
  struct FunctionSummary {
    /// True if this function has reached a max block count while inlined from
    /// at least one call site.
    bool MayReachMaxBlockCount;
    FunctionSummary() : MayReachMaxBlockCount(false) {}
  };

  typedef llvm::DenseMap<const Decl*, FunctionSummary> MapTy;
  MapTy Map;

public:
  void markReachedMaxBlockCount(const Decl* D) {
    Map[D].MayReachMaxBlockCount = true;
  }

  bool hasReachedMaxBlockCount(const Decl* D) {
  MapTy::const_iterator I = Map.find(D);
    if (I != Map.end())
      return I->second.MayReachMaxBlockCount;
    return false;
  }

};

class AnalysisManager : public BugReporterData {
  virtual void anchor();
  AnalysisDeclContextManager AnaCtxMgr;

  ASTContext &Ctx;
  DiagnosticsEngine &Diags;
  const LangOptions &LangOpts;

  OwningPtr<PathDiagnosticConsumer> PD;

  // Configurable components creators.
  StoreManagerCreator CreateStoreMgr;
  ConstraintManagerCreator CreateConstraintMgr;

  CheckerManager *CheckerMgr;

  /// \brief Provide function definitions in other translation units. This is
  /// NULL if we don't have multiple translation units. AnalysisManager does
  /// not own the Indexer.
  idx::Indexer *Idxer;

  enum AnalysisScope { ScopeTU, ScopeDecl } AScope;

  /// \brief The maximum number of exploded nodes the analyzer will generate.
  unsigned MaxNodes;

  /// \brief The maximum number of times the analyzer visits a block.
  unsigned MaxVisit;

  bool VisualizeEGDot;
  bool VisualizeEGUbi;
  AnalysisPurgeMode PurgeDead;

  /// \brief The flag regulates if we should eagerly assume evaluations of
  /// conditionals, thus, bifurcating the path.
  ///
  /// EagerlyAssume - A flag indicating how the engine should handle
  ///   expressions such as: 'x = (y != 0)'.  When this flag is true then
  ///   the subexpression 'y != 0' will be eagerly assumed to be true or false,
  ///   thus evaluating it to the integers 0 or 1 respectively.  The upside
  ///   is that this can increase analysis precision until we have a better way
  ///   to lazily evaluate such logic.  The downside is that it eagerly
  ///   bifurcates paths.
  bool EagerlyAssume;
  bool TrimGraph;
  bool EagerlyTrimEGraph;

public:
  // \brief inter-procedural analysis mode.
  AnalysisIPAMode IPAMode;

  // Settings for inlining tuning.
  /// \brief The inlining stack depth limit.
  unsigned InlineMaxStackDepth;
  /// \brief The max number of basic blocks in a function being inlined.
  unsigned InlineMaxFunctionSize;
  /// \brief The mode of function selection used during inlining.
  AnalysisInliningMode InliningMode;

  /// \brief Do not re-analyze paths leading to exhausted nodes with a different
  /// strategy. We get better code coverage when retry is enabled.
  bool NoRetryExhausted;

public:
  AnalysisManager(ASTContext &ctx, DiagnosticsEngine &diags, 
                  const LangOptions &lang, PathDiagnosticConsumer *pd,
                  StoreManagerCreator storemgr,
                  ConstraintManagerCreator constraintmgr, 
                  CheckerManager *checkerMgr,
                  idx::Indexer *idxer,
                  unsigned maxnodes, unsigned maxvisit,
                  bool vizdot, bool vizubi, AnalysisPurgeMode purge,
                  bool eager, bool trim,
                  bool useUnoptimizedCFG,
                  bool addImplicitDtors, bool addInitializers,
                  bool eagerlyTrimEGraph,
                  AnalysisIPAMode ipa,
                  unsigned inlineMaxStack,
                  unsigned inlineMaxFunctionSize,
                  AnalysisInliningMode inliningMode,
                  bool NoRetry);

  /// Construct a clone of the given AnalysisManager with the given ASTContext
  /// and DiagnosticsEngine.
  AnalysisManager(ASTContext &ctx, DiagnosticsEngine &diags,
                  AnalysisManager &ParentAM);

  ~AnalysisManager() { FlushDiagnostics(); }
  
  void ClearContexts() {
    AnaCtxMgr.clear();
  }
  
  AnalysisDeclContextManager& getAnalysisDeclContextManager() {
    return AnaCtxMgr;
  }

  StoreManagerCreator getStoreManagerCreator() {
    return CreateStoreMgr;
  }

  ConstraintManagerCreator getConstraintManagerCreator() {
    return CreateConstraintMgr;
  }

  CheckerManager *getCheckerManager() const { return CheckerMgr; }

  idx::Indexer *getIndexer() const { return Idxer; }

  virtual ASTContext &getASTContext() {
    return Ctx;
  }

  virtual SourceManager &getSourceManager() {
    return getASTContext().getSourceManager();
  }

  virtual DiagnosticsEngine &getDiagnostic() {
    return Diags;
  }

  const LangOptions &getLangOpts() const {
    return LangOpts;
  }

  virtual PathDiagnosticConsumer *getPathDiagnosticConsumer() {
    return PD.get();
  }
  
  void FlushDiagnostics() {
    if (PD.get())
      PD->FlushDiagnostics(0);
  }

  unsigned getMaxNodes() const { return MaxNodes; }

  unsigned getMaxVisit() const { return MaxVisit; }

  bool shouldVisualizeGraphviz() const { return VisualizeEGDot; }

  bool shouldVisualizeUbigraph() const { return VisualizeEGUbi; }

  bool shouldVisualize() const {
    return VisualizeEGDot || VisualizeEGUbi;
  }

  bool shouldEagerlyTrimExplodedGraph() const { return EagerlyTrimEGraph; }

  bool shouldTrimGraph() const { return TrimGraph; }

  AnalysisPurgeMode getPurgeMode() const { return PurgeDead; }

  bool shouldEagerlyAssume() const { return EagerlyAssume; }

  bool shouldInlineCall() const { return (IPAMode == Inlining); }

  bool hasIndexer() const { return Idxer != 0; }

  AnalysisDeclContext *getAnalysisDeclContextInAnotherTU(const Decl *D);

  CFG *getCFG(Decl const *D) {
    return AnaCtxMgr.getContext(D)->getCFG();
  }

  template <typename T>
  T *getAnalysis(Decl const *D) {
    return AnaCtxMgr.getContext(D)->getAnalysis<T>();
  }

  ParentMap &getParentMap(Decl const *D) {
    return AnaCtxMgr.getContext(D)->getParentMap();
  }

  AnalysisDeclContext *getAnalysisDeclContext(const Decl *D) {
    return AnaCtxMgr.getContext(D);
  }

  AnalysisDeclContext *getAnalysisDeclContext(const Decl *D, idx::TranslationUnit *TU) {
    return AnaCtxMgr.getContext(D, TU);
  }

};

} // enAnaCtxMgrspace

} // end clang namespace

#endif
