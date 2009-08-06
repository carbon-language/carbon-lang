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
  GRStmtNodeBuilder<GRState> &B;
  GRExprEngine &Eng;
  ExplodedNode *Pred;
  SaveAndRestore<bool> OldSink;
  SaveAndRestore<const void*> OldTag;
  SaveAndRestore<ProgramPoint::Kind> OldPointKind;
  SaveOr OldHasGen;

public:
  CheckerContext(ExplodedNodeSet &dst,
                 GRStmtNodeBuilder<GRState> &builder,
                 GRExprEngine &eng,
                 ExplodedNode *pred,
                 const void *tag, bool preVisit)
    : Dst(dst), B(builder), Eng(eng), Pred(pred), 
      OldSink(B.BuildSinks), OldTag(B.Tag),
      OldPointKind(B.PointKind), OldHasGen(B.HasGeneratedNode) {
        assert(Dst.empty());
        B.Tag = tag;
        if (preVisit)
          B.PointKind = ProgramPoint::PreStmtKind;        
      }
  
  ~CheckerContext() {
    if (!B.BuildSinks && Dst.empty() && !B.HasGeneratedNode)
      Dst.Add(Pred);
  }
  
  ConstraintManager &getConstraintManager() {
      return Eng.getConstraintManager();
  }
  ExplodedNodeSet &getNodeSet() { return Dst; }
  GRStmtNodeBuilder<GRState> &getNodeBuilder() { return B; }
  ExplodedNode *&getPredecessor() { return Pred; }
  const GRState *getState() { return B.GetState(Pred); }
  
  ExplodedNode *generateNode(const Stmt* S,
                                      const GRState *state) {
    return B.generateNode(S, state, Pred);
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

  void GR_Visit(ExplodedNodeSet &Dst,
                GRStmtNodeBuilder<GRState> &Builder,
                GRExprEngine &Eng,
                const Stmt *stmt,
                  ExplodedNode *Pred, bool isPrevisit) {
    CheckerContext C(Dst, Builder, Eng, Pred, getTag(), isPrevisit);    
    assert(isPrevisit && "Only previsit supported for now.");
    _PreVisit(C, stmt);
  }
  
public:
  virtual ~Checker() {}
  virtual void _PreVisit(CheckerContext &C, const Stmt *stmt) = 0;
  virtual const void *getTag() = 0;
};

} // end clang namespace

#endif
  
