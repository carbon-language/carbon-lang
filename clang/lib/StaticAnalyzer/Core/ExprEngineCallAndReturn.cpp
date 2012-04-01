//=-- ExprEngineCallAndReturn.cpp - Support for call/return -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines ExprEngine's support for calls and returns.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ObjCMessage.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace clang;
using namespace ento;

void ExprEngine::processCallEnter(CallEnter CE, ExplodedNode *Pred) {
  // Get the entry block in the CFG of the callee.
  const StackFrameContext *calleeCtx = CE.getCalleeContext();
  const CFG *CalleeCFG = calleeCtx->getCFG();
  const CFGBlock *Entry = &(CalleeCFG->getEntry());
  
  // Validate the CFG.
  assert(Entry->empty());
  assert(Entry->succ_size() == 1);
  
  // Get the solitary sucessor.
  const CFGBlock *Succ = *(Entry->succ_begin());
  
  // Construct an edge representing the starting location in the callee.
  BlockEdge Loc(Entry, Succ, calleeCtx);

  // Construct a new state which contains the mapping from actual to
  // formal arguments.
  const LocationContext *callerCtx = Pred->getLocationContext();
  ProgramStateRef state = Pred->getState()->enterStackFrame(callerCtx,
                                                                calleeCtx);
  
  // Construct a new node and add it to the worklist.
  bool isNew;
  ExplodedNode *Node = G.getNode(Loc, state, false, &isNew);
  Node->addPredecessor(Pred, G);
  if (isNew)
    Engine.getWorkList()->enqueue(Node);
}

static const ReturnStmt *getReturnStmt(const ExplodedNode *Node) {
  while (Node) {
    const ProgramPoint &PP = Node->getLocation();
    // Skip any BlockEdges.
    if (isa<BlockEdge>(PP) || isa<CallExit>(PP)) {
      assert(Node->pred_size() == 1);
      Node = *Node->pred_begin();
      continue;
    } 
    if (const StmtPoint *SP = dyn_cast<StmtPoint>(&PP)) {
      const Stmt *S = SP->getStmt();
      return dyn_cast<ReturnStmt>(S);
    }
    break;
  }
  return 0;
}

void ExprEngine::processCallExit(ExplodedNode *Pred) {
  ProgramStateRef state = Pred->getState();
  const StackFrameContext *calleeCtx = 
    Pred->getLocationContext()->getCurrentStackFrame();
  const LocationContext *callerCtx = calleeCtx->getParent();
  const Stmt *CE = calleeCtx->getCallSite();
  
  // If the callee returns an expression, bind its value to CallExpr.
  if (const ReturnStmt *RS = getReturnStmt(Pred)) {
    const LocationContext *LCtx = Pred->getLocationContext();
    SVal V = state->getSVal(RS, LCtx);
    state = state->BindExpr(CE, callerCtx, V);
  }
  
  // Bind the constructed object value to CXXConstructExpr.
  if (const CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(CE)) {
    const CXXThisRegion *ThisR =
    getCXXThisRegion(CCE->getConstructor()->getParent(), calleeCtx);
    
    SVal ThisV = state->getSVal(ThisR);
    // Always bind the region to the CXXConstructExpr.
    state = state->BindExpr(CCE, Pred->getLocationContext(), ThisV);
  }
  
  static SimpleProgramPointTag returnTag("ExprEngine : Call Return");
  PostStmt Loc(CE, callerCtx, &returnTag);
  bool isNew;
  ExplodedNode *N = G.getNode(Loc, state, false, &isNew);
  N->addPredecessor(Pred, G);
  if (!isNew)
    return;
  
  // Perform the post-condition check of the CallExpr.
  ExplodedNodeSet Dst;
  NodeBuilderContext Ctx(Engine, calleeCtx->getCallSiteBlock(), N);
  SaveAndRestore<const NodeBuilderContext*> NBCSave(currentBuilderContext,
                                                    &Ctx);
  SaveAndRestore<unsigned> CBISave(currentStmtIdx, calleeCtx->getIndex());
  
  getCheckerManager().runCheckersForPostStmt(Dst, N, CE, *this,
                                             /* wasInlined */ true);
  
  // Enqueue the next element in the block.
  for (ExplodedNodeSet::iterator I = Dst.begin(), E = Dst.end(); I != E; ++I) {
    Engine.getWorkList()->enqueue(*I,
                                  calleeCtx->getCallSiteBlock(),
                                  calleeCtx->getIndex()+1);
  }
}

static unsigned getNumberStackFrames(const LocationContext *LCtx) {
  unsigned count = 0;
  while (LCtx) {
    if (isa<StackFrameContext>(LCtx))
      ++count;
    LCtx = LCtx->getParent();
  }
  return count;  
}

// Determine if we should inline the call.
bool ExprEngine::shouldInlineDecl(const FunctionDecl *FD, ExplodedNode *Pred) {
  AnalysisDeclContext *CalleeADC = AMgr.getAnalysisDeclContext(FD);
  const CFG *CalleeCFG = CalleeADC->getCFG();

  if (getNumberStackFrames(Pred->getLocationContext())
        == AMgr.InlineMaxStackDepth)
    return false;

  if (FunctionSummaries->hasReachedMaxBlockCount(FD))
    return false;

  if (CalleeCFG->getNumBlockIDs() > AMgr.InlineMaxFunctionSize)
    return false;

  return true;
}

// For now, skip inlining variadic functions.
// We also don't inline blocks.
static bool shouldInlineCallExpr(const CallExpr *CE, ExprEngine *E) {
  if (!E->getAnalysisManager().shouldInlineCall())
    return false;
  QualType callee = CE->getCallee()->getType();
  const FunctionProtoType *FT = 0;
  if (const PointerType *PT = callee->getAs<PointerType>())
    FT = dyn_cast<FunctionProtoType>(PT->getPointeeType());
  else if (const BlockPointerType *BT = callee->getAs<BlockPointerType>()) {
    // FIXME: inline blocks.
    // FT = dyn_cast<FunctionProtoType>(BT->getPointeeType());
    (void) BT;
    return false;
  }
  // If we have no prototype, assume the function is okay.
  if (!FT)
    return true;

  // Skip inlining of variadic functions.
  return !FT->isVariadic();
}

bool ExprEngine::InlineCall(ExplodedNodeSet &Dst,
                            const CallExpr *CE, 
                            ExplodedNode *Pred) {
  if (!shouldInlineCallExpr(CE, this))
    return false;

  ProgramStateRef state = Pred->getState();
  const Expr *Callee = CE->getCallee();
  const FunctionDecl *FD =
    state->getSVal(Callee, Pred->getLocationContext()).getAsFunctionDecl();
  if (!FD || !FD->hasBody(FD))
    return false;
  
  switch (CE->getStmtClass()) {
    default:
      // FIXME: Handle C++.
      break;
    case Stmt::CallExprClass: {
      if (!shouldInlineDecl(FD, Pred))
        return false;

      // Construct a new stack frame for the callee.
      AnalysisDeclContext *CalleeADC = AMgr.getAnalysisDeclContext(FD);
      const StackFrameContext *CallerSFC =
      Pred->getLocationContext()->getCurrentStackFrame();
      const StackFrameContext *CalleeSFC =
      CalleeADC->getStackFrame(CallerSFC, CE,
                               currentBuilderContext->getBlock(),
                               currentStmtIdx);
      
      CallEnter Loc(CE, CalleeSFC, Pred->getLocationContext());
      bool isNew;
      ExplodedNode *N = G.getNode(Loc, state, false, &isNew);
      N->addPredecessor(Pred, G);
      if (isNew)
        Engine.getWorkList()->enqueue(N);
      return true;
    }
  }
  return false;
}

static bool isPointerToConst(const ParmVarDecl *ParamDecl) {
  QualType PointeeTy = ParamDecl->getOriginalType()->getPointeeType();
  if (PointeeTy != QualType() && PointeeTy.isConstQualified() &&
      !PointeeTy->isAnyPointerType() && !PointeeTy->isReferenceType()) {
    return true;
  }
  return false;
}

// Try to retrieve the function declaration and find the function parameter
// types which are pointers/references to a non-pointer const.
// We do not invalidate the corresponding argument regions.
static void findPtrToConstParams(llvm::SmallSet<unsigned, 1> &PreserveArgs,
                       const CallOrObjCMessage &Call) {
  const Decl *CallDecl = Call.getDecl();
  if (!CallDecl)
    return;

  if (const FunctionDecl *FDecl = dyn_cast<FunctionDecl>(CallDecl)) {
    const IdentifierInfo *II = FDecl->getIdentifier();

    // List the cases, where the region should be invalidated even if the
    // argument is const.
    if (II) {
      StringRef FName = II->getName();
      //  - 'int pthread_setspecific(ptheread_key k, const void *)' stores a
      // value into thread local storage. The value can later be retrieved with
      // 'void *ptheread_getspecific(pthread_key)'. So even thought the
      // parameter is 'const void *', the region escapes through the call.
      //  - funopen - sets a buffer for future IO calls.
      //  - ObjC functions that end with "NoCopy" can free memory, of the passed
      // in buffer.
      // - Many CF containers allow objects to escape through custom
      // allocators/deallocators upon container construction.
      // - NSXXInsertXX, for example NSMapInsertIfAbsent, since they can
      // be deallocated by NSMapRemove.
      if (FName == "pthread_setspecific" ||
          FName == "funopen" ||
          FName.endswith("NoCopy") ||
          (FName.startswith("NS") &&
            (FName.find("Insert") != StringRef::npos)) ||
          Call.isCFCGAllowingEscape(FName))
        return;
    }

    for (unsigned Idx = 0, E = Call.getNumArgs(); Idx != E; ++Idx) {
      if (FDecl && Idx < FDecl->getNumParams()) {
        if (isPointerToConst(FDecl->getParamDecl(Idx)))
          PreserveArgs.insert(Idx);
      }
    }
    return;
  }

  if (const ObjCMethodDecl *MDecl = dyn_cast<ObjCMethodDecl>(CallDecl)) {
    assert(MDecl->param_size() <= Call.getNumArgs());
    unsigned Idx = 0;
    for (clang::ObjCMethodDecl::param_const_iterator
         I = MDecl->param_begin(), E = MDecl->param_end(); I != E; ++I, ++Idx) {
      if (isPointerToConst(*I))
        PreserveArgs.insert(Idx);
    }
    return;
  }
}

ProgramStateRef 
ExprEngine::invalidateArguments(ProgramStateRef State,
                                const CallOrObjCMessage &Call,
                                const LocationContext *LC) {
  SmallVector<const MemRegion *, 8> RegionsToInvalidate;

  if (Call.isObjCMessage()) {
    // Invalidate all instance variables of the receiver of an ObjC message.
    // FIXME: We should be able to do better with inter-procedural analysis.
    if (const MemRegion *MR = Call.getInstanceMessageReceiver(LC).getAsRegion())
      RegionsToInvalidate.push_back(MR);

  } else if (Call.isCXXCall()) {
    // Invalidate all instance variables for the callee of a C++ method call.
    // FIXME: We should be able to do better with inter-procedural analysis.
    // FIXME: We can probably do better for const versus non-const methods.
    if (const MemRegion *Callee = Call.getCXXCallee().getAsRegion())
      RegionsToInvalidate.push_back(Callee);

  } else if (Call.isFunctionCall()) {
    // Block calls invalidate all captured-by-reference values.
    SVal CalleeVal = Call.getFunctionCallee();
    if (const MemRegion *Callee = CalleeVal.getAsRegion()) {
      if (isa<BlockDataRegion>(Callee))
        RegionsToInvalidate.push_back(Callee);
    }
  }

  // Indexes of arguments whose values will be preserved by the call.
  llvm::SmallSet<unsigned, 1> PreserveArgs;
  findPtrToConstParams(PreserveArgs, Call);

  for (unsigned idx = 0, e = Call.getNumArgs(); idx != e; ++idx) {
    if (PreserveArgs.count(idx))
      continue;

    SVal V = Call.getArgSVal(idx);

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
  StoreManager::InvalidatedSymbols IS;

  // NOTE: Even if RegionsToInvalidate is empty, we may still invalidate
  //  global variables.
  return State->invalidateRegions(RegionsToInvalidate,
                                  Call.getOriginExpr(), Count, LC,
                                  &IS, &Call);

}

static ProgramStateRef getReplayWithoutInliningState(ExplodedNode *&N,
                                                     const CallExpr *CE) {
  void *ReplayState = N->getState()->get<ReplayWithoutInlining>();
  if (!ReplayState)
    return 0;
  const CallExpr *ReplayCE = reinterpret_cast<const CallExpr*>(ReplayState);
  if (CE == ReplayCE) {
    return N->getState()->remove<ReplayWithoutInlining>();
  }
  return 0;
}

void ExprEngine::VisitCallExpr(const CallExpr *CE, ExplodedNode *Pred,
                               ExplodedNodeSet &dst) {
  // Perform the previsit of the CallExpr.
  ExplodedNodeSet dstPreVisit;
  getCheckerManager().runCheckersForPreStmt(dstPreVisit, Pred, CE, *this);
  
  // Now evaluate the call itself.
  class DefaultEval : public GraphExpander {
    ExprEngine &Eng;
    const CallExpr *CE;
  public:
    
    DefaultEval(ExprEngine &eng, const CallExpr *ce)
    : Eng(eng), CE(ce) {}
    virtual void expandGraph(ExplodedNodeSet &Dst, ExplodedNode *Pred) {

      ProgramStateRef state = getReplayWithoutInliningState(Pred, CE);

      // First, try to inline the call.
      if (state == 0 && Eng.InlineCall(Dst, CE, Pred))
        return;

      // First handle the return value.
      StmtNodeBuilder Bldr(Pred, Dst, *Eng.currentBuilderContext);

      // Get the callee.
      const Expr *Callee = CE->getCallee()->IgnoreParens();
      if (state == 0)
        state = Pred->getState();
      SVal L = state->getSVal(Callee, Pred->getLocationContext());

      // Figure out the result type. We do this dance to handle references.
      QualType ResultTy;
      if (const FunctionDecl *FD = L.getAsFunctionDecl())
        ResultTy = FD->getResultType();
      else
        ResultTy = CE->getType();

      if (CE->isLValue())
        ResultTy = Eng.getContext().getPointerType(ResultTy);

      // Conjure a symbol value to use as the result.
      SValBuilder &SVB = Eng.getSValBuilder();
      unsigned Count = Eng.currentBuilderContext->getCurrentBlockCount();
      const LocationContext *LCtx = Pred->getLocationContext();
      SVal RetVal = SVB.getConjuredSymbolVal(0, CE, LCtx, ResultTy, Count);

      // Generate a new state with the return value set.
      state = state->BindExpr(CE, LCtx, RetVal);

      // Invalidate the arguments.
      state = Eng.invalidateArguments(state, CallOrObjCMessage(CE, state, LCtx),
                                      LCtx);

      // And make the result node.
      Bldr.generateNode(CE, Pred, state);
    }
  };
  
  // Finally, evaluate the function call.  We try each of the checkers
  // to see if the can evaluate the function call.
  ExplodedNodeSet dstCallEvaluated;
  DefaultEval defEval(*this, CE);
  getCheckerManager().runCheckersForEvalCall(dstCallEvaluated,
                                             dstPreVisit,
                                             CE, *this, &defEval);
  
  // Finally, perform the post-condition check of the CallExpr and store
  // the created nodes in 'Dst'.
  getCheckerManager().runCheckersForPostStmt(dst, dstCallEvaluated, CE,
                                             *this);
}

void ExprEngine::VisitReturnStmt(const ReturnStmt *RS, ExplodedNode *Pred,
                                 ExplodedNodeSet &Dst) {
  
  ExplodedNodeSet dstPreVisit;
  getCheckerManager().runCheckersForPreStmt(dstPreVisit, Pred, RS, *this);

  StmtNodeBuilder B(dstPreVisit, Dst, *currentBuilderContext);
  
  if (RS->getRetValue()) {
    for (ExplodedNodeSet::iterator it = dstPreVisit.begin(),
                                  ei = dstPreVisit.end(); it != ei; ++it) {
      B.generateNode(RS, *it, (*it)->getState());
    }
  }
}
