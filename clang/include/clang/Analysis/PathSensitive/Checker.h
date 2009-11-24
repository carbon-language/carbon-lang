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
#include "clang/Analysis/PathSensitive/GRCoreEngine.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"

//===----------------------------------------------------------------------===//
// Checker interface.
//===----------------------------------------------------------------------===//

namespace clang {
  class GRExprEngine;

class CheckerContext {
  ExplodedNodeSet &Dst;
  GRStmtNodeBuilder &B;
  GRExprEngine &Eng;
  ExplodedNode *Pred;
  SaveAndRestore<bool> OldSink;
  SaveAndRestore<const void*> OldTag;
  SaveAndRestore<ProgramPoint::Kind> OldPointKind;
  SaveOr OldHasGen;
  const GRState *state;
  const Stmt *statement;
  const unsigned size;
  bool DoneEvaluating; // FIXME: This is not a permanent API change.
public:
  CheckerContext(ExplodedNodeSet &dst, GRStmtNodeBuilder &builder,
                 GRExprEngine &eng, ExplodedNode *pred,
                 const void *tag, ProgramPoint::Kind K,
                 const Stmt *stmt = 0, const GRState *st = 0)
    : Dst(dst), B(builder), Eng(eng), Pred(pred),
      OldSink(B.BuildSinks),
      OldTag(B.Tag, tag),
      OldPointKind(B.PointKind, K),
      OldHasGen(B.HasGeneratedNode),
      state(st), statement(stmt), size(Dst.size()),
      DoneEvaluating(false) {}

  ~CheckerContext();
  
  // FIXME: This were added to support CallAndMessageChecker to indicating
  // to GRExprEngine to "stop evaluating" a message expression under certain
  // cases.  This is *not* meant to be a permanent API change, and was added
  // to aid in the transition of removing logic for checks from GRExprEngine.  
  void setDoneEvaluating() {
    DoneEvaluating = true;
  }
  bool isDoneEvaluating() const {
    return DoneEvaluating;
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
  const GRState *getState() { return state ? state : B.GetState(Pred); }

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
    if (state != getState() || 
        (state && state != B.GetState(Pred)))
      GenerateNode(state, true);
    else
      Dst.Add(Pred);
  }

  void EmitReport(BugReport *R) {
    Eng.getBugReporter().EmitReport(R);
  }

private:
  ExplodedNode *GenerateNodeImpl(const Stmt* stmt, const GRState *state,
                             bool markAsSink) {
    ExplodedNode *node = B.generateNode(stmt, state, Pred);
    if (markAsSink && node)
      node->markAsSink();
    return node;
  }
  
};

class Checker {
private:
  friend class GRExprEngine;

  // FIXME: Remove the 'tag' option.
  bool GR_Visit(ExplodedNodeSet &Dst,
                GRStmtNodeBuilder &Builder,
                GRExprEngine &Eng,
                const Stmt *S,
                ExplodedNode *Pred, void *tag, bool isPrevisit) {
    CheckerContext C(Dst, Builder, Eng, Pred, tag,
                     isPrevisit ? ProgramPoint::PreStmtKind :
                     ProgramPoint::PostStmtKind, S);
    if (isPrevisit)
      _PreVisit(C, S);
    else
      _PostVisit(C, S);
    return C.isDoneEvaluating();
  }

  // FIXME: Remove the 'tag' option.
  void GR_VisitBind(ExplodedNodeSet &Dst,
                    GRStmtNodeBuilder &Builder, GRExprEngine &Eng,
                    const Stmt *AssignE,
                    const Stmt *StoreE, ExplodedNode *Pred, void *tag, 
                    SVal location, SVal val,
                    bool isPrevisit) {
    CheckerContext C(Dst, Builder, Eng, Pred, tag,
                     isPrevisit ? ProgramPoint::PreStmtKind :
                     ProgramPoint::PostStmtKind, StoreE);
    assert(isPrevisit && "Only previsit supported for now.");
    PreVisitBind(C, AssignE, StoreE, location, val);
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
                     ProgramPoint::PreStoreKind, S, state);
    VisitLocation(C, S, location);
  }

  void GR_EvalDeadSymbols(ExplodedNodeSet &Dst, GRStmtNodeBuilder &Builder,
                          GRExprEngine &Eng, const Stmt *S, ExplodedNode *Pred,
                          SymbolReaper &SymReaper, void *tag) {
    CheckerContext C(Dst, Builder, Eng, Pred, tag, 
                     ProgramPoint::PostPurgeDeadSymbolsKind, S);
    EvalDeadSymbols(C, S, SymReaper);
  }

public:
  virtual ~Checker();
  virtual void _PreVisit(CheckerContext &C, const Stmt *S) {}
  virtual void _PostVisit(CheckerContext &C, const Stmt *S) {}
  virtual void VisitLocation(CheckerContext &C, const Stmt *S, SVal location) {}
  virtual void PreVisitBind(CheckerContext &C, const Stmt *AssignE,
                            const Stmt *StoreE, SVal location, SVal val) {}
  virtual void EvalDeadSymbols(CheckerContext &C, const Stmt *S,
                               SymbolReaper &SymReaper) {}
  virtual void EvalEndPath(GREndPathNodeBuilder &B, void *tag,
                           GRExprEngine &Eng) {}

  virtual void VisitBranchCondition(GRBranchNodeBuilder &Builder,
                                    GRExprEngine &Eng,
                                    Stmt *Condition, void *tag) {}
};
} // end clang namespace

#endif

