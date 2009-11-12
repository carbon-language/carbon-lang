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

public:
  CheckerContext(ExplodedNodeSet &dst, GRStmtNodeBuilder &builder,
                 GRExprEngine &eng, ExplodedNode *pred,
                 const void *tag, ProgramPoint::Kind K,
                 const GRState *st = 0)
    : Dst(dst), B(builder), Eng(eng), Pred(pred),
      OldSink(B.BuildSinks),
      OldTag(B.Tag, tag),
      OldPointKind(B.PointKind, K),
      OldHasGen(B.HasGeneratedNode),
      state(st) {}

  ~CheckerContext() {
    if (!B.BuildSinks && !B.HasGeneratedNode)
      Dst.Add(Pred);
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

  ExplodedNode *GenerateNode(const Stmt *S, bool markAsSink = false) {
    return GenerateNode(S, getState(), markAsSink);
  }

  ExplodedNode *GenerateNode(const Stmt* S, const GRState *state,
                             bool markAsSink = false) {
    ExplodedNode *node = B.generateNode(S, state, Pred);

    if (markAsSink && node)
      node->markAsSink();

    return node;
  }

  void addTransition(ExplodedNode *node) {
    Dst.Add(node);
  }

  void EmitReport(BugReport *R) {
    Eng.getBugReporter().EmitReport(R);
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
                ExplodedNode *Pred, void *tag, bool isPrevisit) {
    CheckerContext C(Dst, Builder, Eng, Pred, tag,
                     isPrevisit ? ProgramPoint::PreStmtKind :
                     ProgramPoint::PostStmtKind);
    if (isPrevisit)
      _PreVisit(C, S);
    else
      _PostVisit(C, S);
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
                     ProgramPoint::PostStmtKind);
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
                     ProgramPoint::PreStoreKind, state);
    VisitLocation(C, S, location);
  }

public:
  virtual ~Checker() {}
  virtual void _PreVisit(CheckerContext &C, const Stmt *S) {}
  virtual void _PostVisit(CheckerContext &C, const Stmt *S) {}
  virtual void VisitLocation(CheckerContext &C, const Stmt *S, SVal location) {}
  virtual void PreVisitBind(CheckerContext &C, const Stmt *AssignE,
                            const Stmt *StoreE, SVal location, SVal val) {}
};
} // end clang namespace

#endif

