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
#include "clang/StaticAnalyzer/Core/PathSensitive/ObjCMessage.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/StmtCXX.h"

using namespace clang;
using namespace ento;

const CXXThisRegion *ExprEngine::getCXXThisRegion(const CXXRecordDecl *D,
                                                 const StackFrameContext *SFC) {
  const Type *T = D->getTypeForDecl();
  QualType PT = getContext().getPointerType(QualType(T, 0));
  return svalBuilder.getRegionManager().getCXXThisRegion(PT, SFC);
}

const CXXThisRegion *ExprEngine::getCXXThisRegion(const CXXMethodDecl *decl,
                                            const StackFrameContext *frameCtx) {
  return svalBuilder.getRegionManager().
                    getCXXThisRegion(decl->getThisType(getContext()), frameCtx);
}

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

void ExprEngine::VisitCXXTemporaryObjectExpr(const CXXTemporaryObjectExpr *expr,
                                             ExplodedNode *Pred,
                                             ExplodedNodeSet &Dst) {
  VisitCXXConstructExpr(expr, 0, Pred, Dst);
}

void ExprEngine::VisitCXXConstructExpr(const CXXConstructExpr *E, 
                                       const MemRegion *Dest,
                                       ExplodedNode *Pred,
                                       ExplodedNodeSet &destNodes) {

#if 0
  const CXXConstructorDecl *CD = E->getConstructor();
  assert(CD);
#endif
  
#if 0
  if (!(CD->doesThisDeclarationHaveABody() && AMgr.shouldInlineCall()))
    // FIXME: invalidate the object.
    return;
#endif
  
#if 0
  // Is the constructor elidable?
  if (E->isElidable()) {
    destNodes.Add(Pred);
    return;
  }
#endif
  
  // Perform the previsit of the constructor.
  ExplodedNodeSet SrcNodes;
  SrcNodes.Add(Pred);
  ExplodedNodeSet TmpNodes;
  getCheckerManager().runCheckersForPreStmt(TmpNodes, SrcNodes, E, *this);
  
  // Evaluate the constructor.  Currently we don't now allow checker-specific
  // implementations of specific constructors (as we do with ordinary
  // function calls.  We can re-evaluate this in the future.
  
#if 0
  // Inlining currently isn't fully implemented.

  if (AMgr.shouldInlineCall()) {
    if (!Dest)
      Dest =
        svalBuilder.getRegionManager().getCXXTempObjectRegion(E,
                                                  Pred->getLocationContext());

    // The callee stack frame context used to create the 'this'
    // parameter region.
    const StackFrameContext *SFC = 
      AMgr.getStackFrame(CD, Pred->getLocationContext(),
                         E, currentBuilderContext->getBlock(),
                         currentStmtIdx);

    // Create the 'this' region.
    const CXXThisRegion *ThisR =
      getCXXThisRegion(E->getConstructor()->getParent(), SFC);

    CallEnter Loc(E, SFC, Pred->getLocationContext());

    StmtNodeBuilder Bldr(SrcNodes, TmpNodes, *currentBuilderContext);
    for (ExplodedNodeSet::iterator NI = SrcNodes.begin(),
                                   NE = SrcNodes.end(); NI != NE; ++NI) {
      ProgramStateRef state = (*NI)->getState();
      // Setup 'this' region, so that the ctor is evaluated on the object pointed
      // by 'Dest'.
      state = state->bindLoc(loc::MemRegionVal(ThisR), loc::MemRegionVal(Dest));
      Bldr.generateNode(Loc, *NI, state);
    }
  }
#endif
  
  // Default semantics: invalidate all regions passed as arguments.
  ExplodedNodeSet destCall;
  {
    StmtNodeBuilder Bldr(TmpNodes, destCall, *currentBuilderContext);
    for (ExplodedNodeSet::iterator i = TmpNodes.begin(), e = TmpNodes.end();
         i != e; ++i)
    {
      ExplodedNode *Pred = *i;
      const LocationContext *LC = Pred->getLocationContext();
      ProgramStateRef state = Pred->getState();

      state = invalidateArguments(state, CallOrObjCMessage(E, state, LC), LC);
      Bldr.generateNode(E, Pred, state);
    }
  }
  // Do the post visit.
  getCheckerManager().runCheckersForPostStmt(destNodes, destCall, E, *this);  
}

void ExprEngine::VisitCXXDestructor(const CXXDestructorDecl *DD,
                                      const MemRegion *Dest,
                                      const Stmt *S,
                                      ExplodedNode *Pred, 
                                      ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currentBuilderContext);
  if (!(DD->doesThisDeclarationHaveABody() && AMgr.shouldInlineCall()))
    return;

  // Create the context for 'this' region.
  const StackFrameContext *SFC =
    AnalysisDeclContexts.getContext(DD)->
      getStackFrame(Pred->getLocationContext(), S,
      currentBuilderContext->getBlock(), currentStmtIdx);

  const CXXThisRegion *ThisR = getCXXThisRegion(DD->getParent(), SFC);

  CallEnter PP(S, SFC, Pred->getLocationContext());

  ProgramStateRef state = Pred->getState();
  state = state->bindLoc(loc::MemRegionVal(ThisR), loc::MemRegionVal(Dest));
  Bldr.generateNode(PP, Pred, state);
}

void ExprEngine::VisitCXXNewExpr(const CXXNewExpr *CNE, ExplodedNode *Pred,
                                   ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currentBuilderContext);
  
  unsigned blockCount = currentBuilderContext->getCurrentBlockCount();
  const LocationContext *LCtx = Pred->getLocationContext();
  DefinedOrUnknownSVal symVal =
    svalBuilder.getConjuredSymbolVal(NULL, CNE, LCtx, CNE->getType(), blockCount);
  const MemRegion *NewReg = cast<loc::MemRegionVal>(symVal).getRegion();  
  QualType ObjTy = CNE->getType()->getAs<PointerType>()->getPointeeType();
  const ElementRegion *EleReg = 
    getStoreManager().GetElementZeroRegion(NewReg, ObjTy);

  if (CNE->isArray()) {
    // FIXME: allocating an array requires simulating the constructors.
    // For now, just return a symbolicated region.
    ProgramStateRef state = Pred->getState();
    state = state->BindExpr(CNE, Pred->getLocationContext(),
                            loc::MemRegionVal(EleReg));
    Bldr.generateNode(CNE, Pred, state);
    return;
  }

  // FIXME: Update for AST changes.
#if 0
  // Evaluate constructor arguments.
  const FunctionProtoType *FnType = NULL;
  const CXXConstructorDecl *CD = CNE->getConstructor();
  if (CD)
    FnType = CD->getType()->getAs<FunctionProtoType>();
  ExplodedNodeSet argsEvaluated;
  Bldr.takeNodes(Pred);
  evalArguments(CNE->constructor_arg_begin(), CNE->constructor_arg_end(),
                FnType, Pred, argsEvaluated);
  Bldr.addNodes(argsEvaluated);

  // Initialize the object region and bind the 'new' expression.
  for (ExplodedNodeSet::iterator I = argsEvaluated.begin(), 
                                 E = argsEvaluated.end(); I != E; ++I) {

    ProgramStateRef state = (*I)->getState();
    
    // Accumulate list of regions that are invalidated.
    // FIXME: Eventually we should unify the logic for constructor
    // processing in one place.
    SmallVector<const MemRegion*, 10> regionsToInvalidate;
    for (CXXNewExpr::const_arg_iterator
          ai = CNE->constructor_arg_begin(), ae = CNE->constructor_arg_end();
          ai != ae; ++ai)
    {
      SVal val = state->getSVal(*ai, (*I)->getLocationContext());
      if (const MemRegion *region = val.getAsRegion())
        regionsToInvalidate.push_back(region);
    }

    if (ObjTy->isRecordType()) {
      regionsToInvalidate.push_back(EleReg);
      // Invalidate the regions.
      // TODO: Pass the call to new information as the last argument, to limit
      // the globals which will get invalidated.
      state = state->invalidateRegions(regionsToInvalidate,
                                       CNE, blockCount, 0, 0);
      
    } else {
      // Invalidate the regions.
      // TODO: Pass the call to new information as the last argument, to limit
      // the globals which will get invalidated.
      state = state->invalidateRegions(regionsToInvalidate,
                                       CNE, blockCount, 0, 0);

      if (CNE->hasInitializer()) {
        SVal V = state->getSVal(*CNE->constructor_arg_begin(),
                                (*I)->getLocationContext());
        state = state->bindLoc(loc::MemRegionVal(EleReg), V);
      } else {
        // Explicitly set to undefined, because currently we retrieve symbolic
        // value from symbolic region.
        state = state->bindLoc(loc::MemRegionVal(EleReg), UndefinedVal());
      }
    }
    state = state->BindExpr(CNE, (*I)->getLocationContext(),
                            loc::MemRegionVal(EleReg));
    Bldr.generateNode(CNE, *I, state);
  }
#endif
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
