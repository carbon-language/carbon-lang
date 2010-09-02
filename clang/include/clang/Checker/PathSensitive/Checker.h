//== Checker.h - Abstract interface for checkers -----------------*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines Checker and CheckerVisitor, classes used for creating
//  domain-specific checks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_CHECKER
#define LLVM_CLANG_ANALYSIS_CHECKER

#include "clang/Analysis/Support/SaveAndRestore.h"
#include "clang/Checker/PathSensitive/GRExprEngine.h"

//===----------------------------------------------------------------------===//
// Checker interface.
//===----------------------------------------------------------------------===//

namespace clang {

class CheckerContext {
  ExplodedNodeSet &Dst;
  GRStmtNodeBuilder &B;
  GRExprEngine &Eng;
  ExplodedNode *Pred;
  SaveAndRestore<bool> OldSink;
  SaveAndRestore<const void*> OldTag;
  SaveAndRestore<ProgramPoint::Kind> OldPointKind;
  SaveOr OldHasGen;
  const GRState *ST;
  const Stmt *statement;
  const unsigned size;
public:
  bool *respondsToCallback;
public:
  CheckerContext(ExplodedNodeSet &dst, GRStmtNodeBuilder &builder,
                 GRExprEngine &eng, ExplodedNode *pred,
                 const void *tag, ProgramPoint::Kind K,
                 bool *respondsToCB = 0,
                 const Stmt *stmt = 0, const GRState *st = 0)
    : Dst(dst), B(builder), Eng(eng), Pred(pred),
      OldSink(B.BuildSinks),
      OldTag(B.Tag, tag),
      OldPointKind(B.PointKind, K),
      OldHasGen(B.HasGeneratedNode),
      ST(st), statement(stmt), size(Dst.size()),
      respondsToCallback(respondsToCB) {}

  ~CheckerContext();

  GRExprEngine &getEngine() {
    return Eng;
  }

  AnalysisManager &getAnalysisManager() {
    return Eng.getAnalysisManager();
  }

  ConstraintManager &getConstraintManager() {
    return Eng.getConstraintManager();
  }

  StoreManager &getStoreManager() {
    return Eng.getStoreManager();
  }

  ExplodedNodeSet &getNodeSet() { return Dst; }
  GRStmtNodeBuilder &getNodeBuilder() { return B; }
  ExplodedNode *&getPredecessor() { return Pred; }
  const GRState *getState() { return ST ? ST : B.GetState(Pred); }

  ASTContext &getASTContext() {
    return Eng.getContext();
  }
  
  BugReporter &getBugReporter() {
    return Eng.getBugReporter();
  }
  
  SourceManager &getSourceManager() {
    return getBugReporter().getSourceManager();
  }

  ValueManager &getValueManager() {
    return Eng.getValueManager();
  }

  SValuator &getSValuator() {
    return Eng.getSValuator();
  }

  ExplodedNode *GenerateNode(bool autoTransition = true) {
    assert(statement && "Only transitions with statements currently supported");
    ExplodedNode *N = GenerateNodeImpl(statement, getState(), false);
    if (N && autoTransition)
      Dst.Add(N);
    return N;
  }
  
  ExplodedNode *GenerateNode(const Stmt *stmt, const GRState *state,
                             bool autoTransition = true) {
    assert(state);
    ExplodedNode *N = GenerateNodeImpl(stmt, state, false);
    if (N && autoTransition)
      addTransition(N);
    return N;
  }

  ExplodedNode *GenerateNode(const GRState *state, ExplodedNode *pred,
                             bool autoTransition = true) {
   assert(statement && "Only transitions with statements currently supported");
    ExplodedNode *N = GenerateNodeImpl(statement, state, pred, false);
    if (N && autoTransition)
      addTransition(N);
    return N;
  }

  ExplodedNode *GenerateNode(const GRState *state, bool autoTransition = true) {
    assert(statement && "Only transitions with statements currently supported");
    ExplodedNode *N = GenerateNodeImpl(statement, state, false);
    if (N && autoTransition)
      addTransition(N);
    return N;
  }

  ExplodedNode *GenerateSink(const Stmt *stmt, const GRState *state = 0) {
    return GenerateNodeImpl(stmt, state ? state : getState(), true);
  }
  
  ExplodedNode *GenerateSink(const GRState *state = 0) {
    assert(statement && "Only transitions with statements currently supported");
    return GenerateNodeImpl(statement, state ? state : getState(), true);
  }

  void addTransition(ExplodedNode *node) {
    Dst.Add(node);
  }
  
  void addTransition(const GRState *state) {
    assert(state);
    // If the 'state' is not new, we need to check if the cached state 'ST'
    // is new.
    if (state != getState() || (ST && ST != B.GetState(Pred)))
      // state is new or equals to ST.
      GenerateNode(state, true);
    else
      Dst.Add(Pred);
  }

  // Generate a node with a new program point different from the one that will
  // be created by the GRStmtNodeBuilder.
  void addTransition(const GRState *state, ProgramPoint Loc) {
    ExplodedNode *N = B.generateNode(Loc, state, Pred);
    if (N)
      addTransition(N);
  }

  void EmitReport(BugReport *R) {
    Eng.getBugReporter().EmitReport(R);
  }

  AnalysisContext *getCurrentAnalysisContext() const {
    return Pred->getLocationContext()->getAnalysisContext();
  }

private:
  ExplodedNode *GenerateNodeImpl(const Stmt* stmt, const GRState *state,
                             bool markAsSink) {
    ExplodedNode *node = B.generateNode(stmt, state, Pred);
    if (markAsSink && node)
      node->markAsSink();
    return node;
  }

  ExplodedNode *GenerateNodeImpl(const Stmt* stmt, const GRState *state,
                                 ExplodedNode *pred, bool markAsSink) {
   ExplodedNode *node = B.generateNode(stmt, state, pred);
    if (markAsSink && node)
      node->markAsSink();
    return node;
  }
};

class Checker {
private:
  friend class GRExprEngine;

  // FIXME: Remove the 'tag' option.
  void GR_Visit(ExplodedNodeSet &Dst,
                GRStmtNodeBuilder &Builder,
                GRExprEngine &Eng,
                const Stmt *S,
                ExplodedNode *Pred, void *tag, bool isPrevisit,
                bool& respondsToCallback) {
    CheckerContext C(Dst, Builder, Eng, Pred, tag,
                     isPrevisit ? ProgramPoint::PreStmtKind :
                     ProgramPoint::PostStmtKind, &respondsToCallback, S);
    if (isPrevisit)
      _PreVisit(C, S);
    else
      _PostVisit(C, S);
  }

  bool GR_EvalNilReceiver(ExplodedNodeSet &Dst, GRStmtNodeBuilder &Builder,
                          GRExprEngine &Eng, const ObjCMessageExpr *ME,
                          ExplodedNode *Pred, const GRState *state, void *tag) {
    CheckerContext C(Dst, Builder, Eng, Pred, tag, ProgramPoint::PostStmtKind,
                     0, ME, state);
    return EvalNilReceiver(C, ME);
  }

  bool GR_EvalCallExpr(ExplodedNodeSet &Dst, GRStmtNodeBuilder &Builder,
                       GRExprEngine &Eng, const CallExpr *CE,
                       ExplodedNode *Pred, void *tag) {
    CheckerContext C(Dst, Builder, Eng, Pred, tag, ProgramPoint::PostStmtKind,
                     0, CE);
    return EvalCallExpr(C, CE);
  }

  // FIXME: Remove the 'tag' option.
  void GR_VisitBind(ExplodedNodeSet &Dst,
                    GRStmtNodeBuilder &Builder, GRExprEngine &Eng,
                    const Stmt *StoreE, ExplodedNode *Pred, void *tag, 
                    SVal location, SVal val,
                    bool isPrevisit) {
    CheckerContext C(Dst, Builder, Eng, Pred, tag,
                     isPrevisit ? ProgramPoint::PreStmtKind :
                     ProgramPoint::PostStmtKind, 0, StoreE);
    assert(isPrevisit && "Only previsit supported for now.");
    PreVisitBind(C, StoreE, location, val);
  }
  
  // FIXME: Remove the 'tag' option.
  void GR_VisitLocation(ExplodedNodeSet &Dst,
                        GRStmtNodeBuilder &Builder,
                        GRExprEngine &Eng,
                        const Stmt *S,
                        ExplodedNode *Pred, const GRState *state,
                        SVal location,
                        void *tag, bool isLoad) {
    CheckerContext C(Dst, Builder, Eng, Pred, tag,
                     isLoad ? ProgramPoint::PreLoadKind :
                     ProgramPoint::PreStoreKind, 0, S, state);
    VisitLocation(C, S, location);
  }

  void GR_EvalDeadSymbols(ExplodedNodeSet &Dst, GRStmtNodeBuilder &Builder,
                          GRExprEngine &Eng, const Stmt *S, ExplodedNode *Pred,
                          SymbolReaper &SymReaper, void *tag) {
    CheckerContext C(Dst, Builder, Eng, Pred, tag, 
                     ProgramPoint::PostPurgeDeadSymbolsKind, 0, S);
    EvalDeadSymbols(C, SymReaper);
  }

public:
  virtual ~Checker();
  virtual void _PreVisit(CheckerContext &C, const Stmt *S) {}
  virtual void _PostVisit(CheckerContext &C, const Stmt *S) {}
  virtual void VisitLocation(CheckerContext &C, const Stmt *S, SVal location) {}
  virtual void PreVisitBind(CheckerContext &C, const Stmt *StoreE,
                            SVal location, SVal val) {}
  virtual void EvalDeadSymbols(CheckerContext &C, SymbolReaper &SymReaper) {}
  virtual void EvalEndPath(GREndPathNodeBuilder &B, void *tag,
                           GRExprEngine &Eng) {}

  virtual void MarkLiveSymbols(const GRState *state, SymbolReaper &SymReaper) {}

  virtual void VisitBranchCondition(GRBranchNodeBuilder &Builder,
                                    GRExprEngine &Eng,
                                    const Stmt *Condition, void *tag) {}

  virtual bool EvalNilReceiver(CheckerContext &C, const ObjCMessageExpr *ME) {
    return false;
  }

  virtual bool EvalCallExpr(CheckerContext &C, const CallExpr *CE) {
    return false;
  }

  virtual const GRState *EvalAssume(const GRState *state, SVal Cond, 
                                    bool Assumption, bool *respondsToCallback) {
    *respondsToCallback = false;
    return state;
  }

  virtual bool WantsRegionChangeUpdate(const GRState *state) { return false; }

  virtual const GRState *EvalRegionChanges(const GRState *state,
                                           const MemRegion * const *Begin,
                                           const MemRegion * const *End,
                                           bool *respondsToCallback) {
    *respondsToCallback = false;
    return state;
  }

  virtual void VisitEndAnalysis(ExplodedGraph &G, BugReporter &B,
                                GRExprEngine &Eng) {}
};
} // end clang namespace

#endif

