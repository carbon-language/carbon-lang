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

  CallEnter PP(S, SFC, Pred->getLocationContext());
  ProgramStateRef state = Pred->getState();
  state = state->bindLoc(svalBuilder.getCXXThis(DD->getParent(), SFC),
                         loc::MemRegionVal(Dest));
  Bldr.generateNode(PP, Pred, state);
}

static bool isPointerToConst(const ParmVarDecl *ParamDecl) {
  // FIXME: Copied from ExprEngineCallAndReturn.cpp
  QualType PointeeTy = ParamDecl->getOriginalType()->getPointeeType();
  if (PointeeTy != QualType() && PointeeTy.isConstQualified() &&
      !PointeeTy->isAnyPointerType() && !PointeeTy->isReferenceType()) {
    return true;
  }
  return false;
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
  ProgramStateRef State = Pred->getState();

  if (CNE->isArray()) {
    // FIXME: allocating an array requires simulating the constructors.
    // For now, just return a symbolicated region.
    State = State->BindExpr(CNE, Pred->getLocationContext(),
                            loc::MemRegionVal(EleReg));
    Bldr.generateNode(CNE, Pred, State);
    return;
  }

  FunctionDecl *FD = CNE->getOperatorNew();
  if (FD && FD->isReservedGlobalPlacementOperator()) {
    // Non-array placement new should always return the placement location.
    SVal PlacementLoc = State->getSVal(CNE->getPlacementArg(0), LCtx);
    State = State->BindExpr(CNE, LCtx, PlacementLoc);
    // FIXME: Once we have proper support for CXXConstructExprs inside
    // CXXNewExpr, we need to make sure that the constructed object is not
    // immediately invalidated here. (The placement call should happen before
    // the constructor call anyway.)
  }

  // Invalidate placement args.

  // FIXME: This is largely copied from invalidateArguments, because
  // CallOrObjCMessage is not general enough to handle new-expressions yet.
  SmallVector<const MemRegion *, 4> RegionsToInvalidate;

  unsigned Index = 0;
  for (CXXNewExpr::const_arg_iterator I = CNE->placement_arg_begin(),
                                      E = CNE->placement_arg_end();
       I != E; ++I) {
    // Pre-increment the argument index to skip over the implicit size arg.
    ++Index;
    if (FD && Index < FD->getNumParams())
      if (isPointerToConst(FD->getParamDecl(Index)))
        continue;
    
    SVal V = State->getSVal(*I, LCtx);
    
    // If we are passing a location wrapped as an integer, unwrap it and
    // invalidate the values referred by the location.
    if (nonloc::LocAsInteger *Wrapped = dyn_cast<nonloc::LocAsInteger>(&V))
      V = Wrapped->getLoc();
    else if (!isa<Loc>(V))
      continue;
    
    if (const MemRegion *R = V.getAsRegion()) {
      // Invalidate the value of the variable passed by reference.
      
      // Are we dealing with an ElementRegion?  If the element type is
      // a basic integer type (e.g., char, int) and the underlying region
      // is a variable region then strip off the ElementRegion.
      // FIXME: We really need to think about this for the general case
      //   as sometimes we are reasoning about arrays and other times
      //   about (char*), etc., is just a form of passing raw bytes.
      //   e.g., void *p = alloca(); foo((char*)p);
      if (const ElementRegion *ER = dyn_cast<ElementRegion>(R)) {
        // Checking for 'integral type' is probably too promiscuous, but
        // we'll leave it in for now until we have a systematic way of
        // handling all of these cases.  Eventually we need to come up
        // with an interface to StoreManager so that this logic can be
        // appropriately delegated to the respective StoreManagers while
        // still allowing us to do checker-specific logic (e.g.,
        // invalidating reference counts), probably via callbacks.
        if (ER->getElementType()->isIntegralOrEnumerationType()) {
          const MemRegion *superReg = ER->getSuperRegion();
          if (isa<VarRegion>(superReg) || isa<FieldRegion>(superReg) ||
              isa<ObjCIvarRegion>(superReg))
            R = cast<TypedRegion>(superReg);
        }
        // FIXME: What about layers of ElementRegions?
      }
      
      // Mark this region for invalidation.  We batch invalidate regions
      // below for efficiency.
      RegionsToInvalidate.push_back(R);
    } else {
      // Nuke all other arguments passed by reference.
      // FIXME: is this necessary or correct? This handles the non-Region
      //  cases.  Is it ever valid to store to these?
      State = State->unbindLoc(cast<Loc>(V));
    }
  }
  
  // Invalidate designated regions using the batch invalidation API.
  
  // FIXME: We can have collisions on the conjured symbol if the
  //  expression *I also creates conjured symbols.  We probably want
  //  to identify conjured symbols by an expression pair: the enclosing
  //  expression (the context) and the expression itself.  This should
  //  disambiguate conjured symbols.
  unsigned Count = currentBuilderContext->getCurrentBlockCount();
  
  // NOTE: Even if RegionsToInvalidate is empty, we may still invalidate
  //  global variables.
  State = State->invalidateRegions(RegionsToInvalidate, CNE, Count, LCtx);
  Bldr.generateNode(CNE, Pred, State);
  return;

  // FIXME: The below code is long-since dead. However, constructor handling
  // in new-expressions is far from complete. See PR12014 for more details.
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
