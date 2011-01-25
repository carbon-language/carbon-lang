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

#ifndef LLVM_CLANG_GR_CHECKER
#define LLVM_CLANG_GR_CHECKER

#include "clang/Analysis/Support/SaveAndRestore.h"
#include "clang/StaticAnalyzer/PathSensitive/ExprEngine.h"

//===----------------------------------------------------------------------===//
// Checker interface.
//===----------------------------------------------------------------------===//

namespace clang {

namespace ento {

class CheckerContext {
  ExplodedNodeSet &Dst;
  StmtNodeBuilder &B;
  ExprEngine &Eng;
  ExplodedNode *Pred;
  SaveAndRestore<bool> OldSink;
  const void *checkerTag;
  SaveAndRestore<ProgramPoint::Kind> OldPointKind;
  SaveOr OldHasGen;
  const GRState *ST;
  const Stmt *statement;
  const unsigned size;
public:
  bool *respondsToCallback;
public:
  CheckerContext(ExplodedNodeSet &dst, StmtNodeBuilder &builder,
                 ExprEngine &eng, ExplodedNode *pred,
                 const void *tag, ProgramPoint::Kind K,
                 bool *respondsToCB = 0,
                 const Stmt *stmt = 0, const GRState *st = 0)
    : Dst(dst), B(builder), Eng(eng), Pred(pred),
      OldSink(B.BuildSinks),
      checkerTag(tag),
      OldPointKind(B.PointKind, K),
      OldHasGen(B.hasGeneratedNode),
      ST(st), statement(stmt), size(Dst.size()),
      respondsToCallback(respondsToCB) {}

  ~CheckerContext();

  ExprEngine &getEngine() {
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
  StmtNodeBuilder &getNodeBuilder() { return B; }
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

  SValBuilder &getSValBuilder() {
    return Eng.getSValBuilder();
  }

  ExplodedNode *generateNode(bool autoTransition = true) {
    assert(statement && "Only transitions with statements currently supported");
    ExplodedNode *N = generateNodeImpl(statement, getState(), false,
                                       checkerTag);
    if (N && autoTransition)
      Dst.Add(N);
    return N;
  }
  
  ExplodedNode *generateNode(const Stmt *stmt, const GRState *state,
                             bool autoTransition = true, const void *tag = 0) {
    assert(state);
    ExplodedNode *N = generateNodeImpl(stmt, state, false,
                                       tag ? tag : checkerTag);
    if (N && autoTransition)
      addTransition(N);
    return N;
  }

  ExplodedNode *generateNode(const GRState *state, ExplodedNode *pred,
                             bool autoTransition = true) {
   assert(statement && "Only transitions with statements currently supported");
    ExplodedNode *N = generateNodeImpl(statement, state, pred, false);
    if (N && autoTransition)
      addTransition(N);
    return N;
  }

  ExplodedNode *generateNode(const GRState *state, bool autoTransition = true,
                             const void *tag = 0) {
    assert(statement && "Only transitions with statements currently supported");
    ExplodedNode *N = generateNodeImpl(statement, state, false,
                                       tag ? tag : checkerTag);
    if (N && autoTransition)
      addTransition(N);
    return N;
  }

  ExplodedNode *generateSink(const Stmt *stmt, const GRState *state = 0) {
    return generateNodeImpl(stmt, state ? state : getState(), true,
                            checkerTag);
  }
  
  ExplodedNode *generateSink(const GRState *state = 0) {
    assert(statement && "Only transitions with statements currently supported");
    return generateNodeImpl(statement, state ? state : getState(), true,
                            checkerTag);
  }

  void addTransition(ExplodedNode *node) {
    Dst.Add(node);
  }
  
  void addTransition(const GRState *state, const void *tag = 0) {
    assert(state);
    // If the 'state' is not new, we need to check if the cached state 'ST'
    // is new.
    if (state != getState() || (ST && ST != B.GetState(Pred)))
      // state is new or equals to ST.
      generateNode(state, true, tag);
    else
      Dst.Add(Pred);
  }

  void EmitReport(BugReport *R) {
    Eng.getBugReporter().EmitReport(R);
  }

  AnalysisContext *getCurrentAnalysisContext() const {
    return Pred->getLocationContext()->getAnalysisContext();
  }

private:
  ExplodedNode *generateNodeImpl(const Stmt* stmt, const GRState *state,
                             bool markAsSink, const void *tag) {
    ExplodedNode *node = B.generateNode(stmt, state, Pred, tag);
    if (markAsSink && node)
      node->markAsSink();
    return node;
  }

  ExplodedNode *generateNodeImpl(const Stmt* stmt, const GRState *state,
                                 ExplodedNode *pred, bool markAsSink) {
   ExplodedNode *node = B.generateNode(stmt, state, pred, checkerTag);
    if (markAsSink && node)
      node->markAsSink();
    return node;
  }
};

class Checker {
private:
  friend class ExprEngine;

  // FIXME: Remove the 'tag' option.
  void GR_Visit(ExplodedNodeSet &Dst,
                StmtNodeBuilder &Builder,
                ExprEngine &Eng,
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

  void GR_visitObjCMessage(ExplodedNodeSet &Dst,
                           StmtNodeBuilder &Builder,
                           ExprEngine &Eng,
                           const ObjCMessage &msg,
                           ExplodedNode *Pred, void *tag, bool isPrevisit) {
    CheckerContext C(Dst, Builder, Eng, Pred, tag,
                     isPrevisit ? ProgramPoint::PreStmtKind :
                     ProgramPoint::PostStmtKind, 0, msg.getOriginExpr());
    if (isPrevisit)
      preVisitObjCMessage(C, msg);
    else
      postVisitObjCMessage(C, msg);
  }

  bool GR_evalNilReceiver(ExplodedNodeSet &Dst, StmtNodeBuilder &Builder,
                          ExprEngine &Eng, const ObjCMessage &msg,
                          ExplodedNode *Pred, const GRState *state, void *tag) {
    CheckerContext C(Dst, Builder, Eng, Pred, tag, ProgramPoint::PostStmtKind,
                     0, msg.getOriginExpr(), state);
    return evalNilReceiver(C, msg);
  }

  bool GR_evalCallExpr(ExplodedNodeSet &Dst, StmtNodeBuilder &Builder,
                       ExprEngine &Eng, const CallExpr *CE,
                       ExplodedNode *Pred, void *tag) {
    CheckerContext C(Dst, Builder, Eng, Pred, tag, ProgramPoint::PostStmtKind,
                     0, CE);
    return evalCallExpr(C, CE);
  }

  // FIXME: Remove the 'tag' option.
  void GR_VisitBind(ExplodedNodeSet &Dst,
                    StmtNodeBuilder &Builder, ExprEngine &Eng,
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
  void GR_visitLocation(ExplodedNodeSet &Dst,
                        StmtNodeBuilder &Builder,
                        ExprEngine &Eng,
                        const Stmt *S,
                        ExplodedNode *Pred, const GRState *state,
                        SVal location,
                        void *tag, bool isLoad) {
    CheckerContext C(Dst, Builder, Eng, Pred, tag,
                     isLoad ? ProgramPoint::PreLoadKind :
                     ProgramPoint::PreStoreKind, 0, S, state);
    visitLocation(C, S, location, isLoad);
  }

  void GR_evalDeadSymbols(ExplodedNodeSet &Dst, StmtNodeBuilder &Builder,
                          ExprEngine &Eng, const Stmt *S, ExplodedNode *Pred,
                          SymbolReaper &SymReaper, void *tag) {
    CheckerContext C(Dst, Builder, Eng, Pred, tag, 
                     ProgramPoint::PostPurgeDeadSymbolsKind, 0, S);
    evalDeadSymbols(C, SymReaper);
  }

public:
  virtual ~Checker();
  virtual void _PreVisit(CheckerContext &C, const Stmt *S) {}
  virtual void _PostVisit(CheckerContext &C, const Stmt *S) {}
  virtual void preVisitObjCMessage(CheckerContext &C, ObjCMessage msg) {}
  virtual void postVisitObjCMessage(CheckerContext &C, ObjCMessage msg) {}
  virtual void visitLocation(CheckerContext &C, const Stmt *S, SVal location,
                             bool isLoad) {}
  virtual void PreVisitBind(CheckerContext &C, const Stmt *StoreE,
                            SVal location, SVal val) {}
  virtual void evalDeadSymbols(CheckerContext &C, SymbolReaper &SymReaper) {}
  virtual void evalEndPath(EndOfFunctionNodeBuilder &B, void *tag,
                           ExprEngine &Eng) {}

  virtual void MarkLiveSymbols(const GRState *state, SymbolReaper &SymReaper) {}

  virtual void VisitBranchCondition(BranchNodeBuilder &Builder,
                                    ExprEngine &Eng,
                                    const Stmt *Condition, void *tag) {}

  virtual bool evalNilReceiver(CheckerContext &C, ObjCMessage msg) {
    return false;
  }

  virtual bool evalCallExpr(CheckerContext &C, const CallExpr *CE) {
    return false;
  }

  virtual const GRState *evalAssume(const GRState *state, SVal Cond, 
                                    bool Assumption, bool *respondsToCallback) {
    *respondsToCallback = false;
    return state;
  }

  virtual bool wantsRegionChangeUpdate(const GRState *state) { return false; }

  virtual const GRState *EvalRegionChanges(const GRState *state,
                                           const MemRegion * const *Begin,
                                           const MemRegion * const *End,
                                           bool *respondsToCallback) {
    *respondsToCallback = false;
    return state;
  }

  virtual void VisitEndAnalysis(ExplodedGraph &G, BugReporter &B,
                                ExprEngine &Eng) {}
};

} // end GR namespace

} // end clang namespace

#endif

