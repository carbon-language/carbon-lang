//===- ExprEngineCXX.cpp - ExprEngine support for C++ -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the C++ expression evaluation engine.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/Calls.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/StmtCXX.h"

using namespace clang;
using namespace ento;

void ExprEngine::CreateCXXTemporaryObject(const MaterializeTemporaryExpr *ME,
                                          ExplodedNode *Pred,
                                          ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currentBuilderContext);
  const Expr *tempExpr = ME->GetTemporaryExpr()->IgnoreParens();
  ProgramStateRef state = Pred->getState();
  const LocationContext *LCtx = Pred->getLocationContext();

  // Bind the temporary object to the value of the expression. Then bind
  // the expression to the location of the object.
  SVal V = state->getSVal(tempExpr, Pred->getLocationContext());

  const MemRegion *R =
    svalBuilder.getRegionManager().getCXXTempObjectRegion(ME, LCtx);

  state = state->bindLoc(loc::MemRegionVal(R), V);
  Bldr.generateNode(ME, Pred, state->BindExpr(ME, LCtx, loc::MemRegionVal(R)));
}

void ExprEngine::VisitCXXConstructExpr(const CXXConstructExpr *CE,
                                       const MemRegion *Dest,
                                       ExplodedNode *Pred,
                                       ExplodedNodeSet &destNodes) {
  CXXConstructorCall Call(CE, Dest, Pred->getState(),
                          Pred->getLocationContext());

  ExplodedNodeSet DstPreVisit;
  getCheckerManager().runCheckersForPreStmt(DstPreVisit, Pred, CE, *this);
  ExplodedNodeSet DstPreCall;
  getCheckerManager().runCheckersForPreCall(DstPreCall, DstPreVisit,
                                            Call, *this);

  ExplodedNodeSet DstInvalidated;
  StmtNodeBuilder Bldr(DstPreCall, DstInvalidated, *currentBuilderContext);
  for (ExplodedNodeSet::iterator I = DstPreCall.begin(), E = DstPreCall.end();
       I != E; ++I)
    defaultEvalCall(Bldr, *I, Call);

  ExplodedNodeSet DstPostCall;
  getCheckerManager().runCheckersForPostCall(DstPostCall, DstInvalidated,
                                             Call, *this);
  getCheckerManager().runCheckersForPostStmt(destNodes, DstPostCall, CE, *this);
}

void ExprEngine::VisitCXXDestructor(const CXXDestructorDecl *DD,
                                      const MemRegion *Dest,
                                      const Stmt *S,
                                      ExplodedNode *Pred, 
                                      ExplodedNodeSet &Dst) {
  CXXDestructorCall Call(DD, S, Dest, Pred->getState(),
                         Pred->getLocationContext());

  ExplodedNodeSet DstPreCall;
  getCheckerManager().runCheckersForPreCall(DstPreCall, Pred,
                                            Call, *this);

  ExplodedNodeSet DstInvalidated;
  StmtNodeBuilder Bldr(DstPreCall, DstInvalidated, *currentBuilderContext);
  for (ExplodedNodeSet::iterator I = DstPreCall.begin(), E = DstPreCall.end();
       I != E; ++I)
    defaultEvalCall(Bldr, *I, Call);

  ExplodedNodeSet DstPostCall;
  getCheckerManager().runCheckersForPostCall(Dst, DstInvalidated,
                                             Call, *this);
}

void ExprEngine::VisitCXXNewExpr(const CXXNewExpr *CNE, ExplodedNode *Pred,
                                   ExplodedNodeSet &Dst) {
  // FIXME: Much of this should eventually migrate to CXXAllocatorCall.
  // Also, we need to decide how allocators actually work -- they're not
  // really part of the CXXNewExpr because they happen BEFORE the
  // CXXConstructExpr subexpression. See PR12014 for some discussion.
  StmtNodeBuilder Bldr(Pred, Dst, *currentBuilderContext);
  
  unsigned blockCount = currentBuilderContext->getCurrentBlockCount();
  const LocationContext *LCtx = Pred->getLocationContext();
  DefinedOrUnknownSVal symVal =
    svalBuilder.getConjuredSymbolVal(0, CNE, LCtx, CNE->getType(), blockCount);
  ProgramStateRef State = Pred->getState();

  // Invalidate placement args.
  CXXAllocatorCall Call(CNE, State, LCtx);
  State = Call.invalidateRegions(blockCount);

  if (CNE->isArray()) {
    // FIXME: allocating an array requires simulating the constructors.
    // For now, just return a symbolicated region.
    const MemRegion *NewReg = cast<loc::MemRegionVal>(symVal).getRegion();
    QualType ObjTy = CNE->getType()->getAs<PointerType>()->getPointeeType();
    const ElementRegion *EleReg =
      getStoreManager().GetElementZeroRegion(NewReg, ObjTy);
    State = State->BindExpr(CNE, Pred->getLocationContext(),
                            loc::MemRegionVal(EleReg));
    Bldr.generateNode(CNE, Pred, State);
    return;
  }

  // FIXME: Once we have proper support for CXXConstructExprs inside
  // CXXNewExpr, we need to make sure that the constructed object is not
  // immediately invalidated here. (The placement call should happen before
  // the constructor call anyway.)
  FunctionDecl *FD = CNE->getOperatorNew();
  if (FD && FD->isReservedGlobalPlacementOperator()) {
    // Non-array placement new should always return the placement location.
    SVal PlacementLoc = State->getSVal(CNE->getPlacementArg(0), LCtx);
    State = State->BindExpr(CNE, LCtx, PlacementLoc);
  } else {
    State = State->BindExpr(CNE, LCtx, symVal);
  }

  // If the type is not a record, we won't have a CXXConstructExpr as an
  // initializer. Copy the value over.
  if (const Expr *Init = CNE->getInitializer()) {
    if (!isa<CXXConstructExpr>(Init)) {
      QualType ObjTy = CNE->getType()->getAs<PointerType>()->getPointeeType();
      (void)ObjTy;
      assert(!ObjTy->isRecordType());
      SVal Location = State->getSVal(CNE, LCtx);
      if (isa<Loc>(Location))
        State = State->bindLoc(cast<Loc>(Location), State->getSVal(Init, LCtx));
    }
  }

  Bldr.generateNode(CNE, Pred, State);
}

void ExprEngine::VisitCXXDeleteExpr(const CXXDeleteExpr *CDE, 
                                    ExplodedNode *Pred, ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currentBuilderContext);
  ProgramStateRef state = Pred->getState();
  Bldr.generateNode(CDE, Pred, state);
}

void ExprEngine::VisitCXXCatchStmt(const CXXCatchStmt *CS,
                                   ExplodedNode *Pred,
                                   ExplodedNodeSet &Dst) {
  const VarDecl *VD = CS->getExceptionDecl();
  if (!VD) {
    Dst.Add(Pred);
    return;
  }

  const LocationContext *LCtx = Pred->getLocationContext();
  SVal V = svalBuilder.getConjuredSymbolVal(CS, LCtx, VD->getType(),
                                 currentBuilderContext->getCurrentBlockCount());
  ProgramStateRef state = Pred->getState();
  state = state->bindLoc(state->getLValue(VD, LCtx), V);

  StmtNodeBuilder Bldr(Pred, Dst, *currentBuilderContext);
  Bldr.generateNode(CS, Pred, state);
}

void ExprEngine::VisitCXXThisExpr(const CXXThisExpr *TE, ExplodedNode *Pred,
                                    ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currentBuilderContext);

  // Get the this object region from StoreManager.
  const LocationContext *LCtx = Pred->getLocationContext();
  const MemRegion *R =
    svalBuilder.getRegionManager().getCXXThisRegion(
                                  getContext().getCanonicalType(TE->getType()),
                                                    LCtx);

  ProgramStateRef state = Pred->getState();
  SVal V = state->getSVal(loc::MemRegionVal(R));
  Bldr.generateNode(TE, Pred, state->BindExpr(TE, LCtx, V));
}
