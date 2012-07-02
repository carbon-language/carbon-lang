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

#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/Calls.h"
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

// Find the last statement on the path to the exploded node and the
// corresponding Block.
static std::pair<const Stmt*,
                 const CFGBlock*> getLastStmt(const ExplodedNode *Node) {
  const Stmt *S = 0;
  const CFGBlock *Blk = 0;
  const StackFrameContext *SF =
          Node->getLocation().getLocationContext()->getCurrentStackFrame();
  while (Node) {
    const ProgramPoint &PP = Node->getLocation();
    // Skip any BlockEdges, empty blocks, and the CallExitBegin node.
    if (isa<BlockEdge>(PP) || isa<CallExitBegin>(PP) || isa<BlockEntrance>(PP)){
      assert(Node->pred_size() == 1);
      Node = *Node->pred_begin();
      continue;
    }
    // If we reached the CallEnter, the function has no statements.
    if (isa<CallEnter>(PP))
      break;
    if (const StmtPoint *SP = dyn_cast<StmtPoint>(&PP)) {
      S = SP->getStmt();
      // Now, get the enclosing basic block.
      while (Node && Node->pred_size() >=1 ) {
        const ProgramPoint &PP = Node->getLocation();
        if (isa<BlockEdge>(PP) &&
            (PP.getLocationContext()->getCurrentStackFrame() == SF)) {
          BlockEdge &EPP = cast<BlockEdge>(PP);
          Blk = EPP.getDst();
          break;
        }
        Node = *Node->pred_begin();
      }
      break;
    }
    break;
  }
  return std::pair<const Stmt*, const CFGBlock*>(S, Blk);
}

/// The call exit is simulated with a sequence of nodes, which occur between 
/// CallExitBegin and CallExitEnd. The following operations occur between the 
/// two program points:
/// 1. CallExitBegin (triggers the start of call exit sequence)
/// 2. Bind the return value
/// 3. Run Remove dead bindings to clean up the dead symbols from the callee.
/// 4. CallExitEnd (switch to the caller context)
/// 5. PostStmt<CallExpr>
void ExprEngine::processCallExit(ExplodedNode *CEBNode) {
  // Step 1 CEBNode was generated before the call.

  const StackFrameContext *calleeCtx =
      CEBNode->getLocationContext()->getCurrentStackFrame();
  
  // The parent context might not be a stack frame, so make sure we
  // look up the first enclosing stack frame.
  const StackFrameContext *callerCtx =
    calleeCtx->getParent()->getCurrentStackFrame();
  
  const Stmt *CE = calleeCtx->getCallSite();
  ProgramStateRef state = CEBNode->getState();
  // Find the last statement in the function and the corresponding basic block.
  const Stmt *LastSt = 0;
  const CFGBlock *Blk = 0;
  llvm::tie(LastSt, Blk) = getLastStmt(CEBNode);

  // Step 2: generate node with binded return value: CEBNode -> BindedRetNode.

  // If the callee returns an expression, bind its value to CallExpr.
  if (const ReturnStmt *RS = dyn_cast_or_null<ReturnStmt>(LastSt)) {
    const LocationContext *LCtx = CEBNode->getLocationContext();
    SVal V = state->getSVal(RS, LCtx);
    state = state->BindExpr(CE, callerCtx, V);
  }

  // Bind the constructed object value to CXXConstructExpr.
  if (const CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(CE)) {
    loc::MemRegionVal This =
      svalBuilder.getCXXThis(CCE->getConstructor()->getParent(), calleeCtx);
    SVal ThisV = state->getSVal(This);

    // Always bind the region to the CXXConstructExpr.
    state = state->BindExpr(CCE, CEBNode->getLocationContext(), ThisV);
  }

  static SimpleProgramPointTag retValBindTag("ExprEngine : Bind Return Value");
  PostStmt Loc(LastSt, calleeCtx, &retValBindTag);
  bool isNew;
  ExplodedNode *BindedRetNode = G.getNode(Loc, state, false, &isNew);
  BindedRetNode->addPredecessor(CEBNode, G);
  if (!isNew)
    return;

  // Step 3: BindedRetNode -> CleanedNodes
  // If we can find a statement and a block in the inlined function, run remove
  // dead bindings before returning from the call. This is important to ensure
  // that we report the issues such as leaks in the stack contexts in which
  // they occurred.
  ExplodedNodeSet CleanedNodes;
  if (LastSt && Blk) {
    NodeBuilderContext Ctx(getCoreEngine(), Blk, BindedRetNode);
    currentBuilderContext = &Ctx;
    // Here, we call the Symbol Reaper with 0 statement and caller location
    // context, telling it to clean up everything in the callee's context
    // (and it's children). We use LastStmt as a diagnostic statement, which
    // which the PreStmtPurge Dead point will be associated.
    removeDead(BindedRetNode, CleanedNodes, 0, callerCtx, LastSt,
               ProgramPoint::PostStmtPurgeDeadSymbolsKind);
    currentBuilderContext = 0;
  } else {
    CleanedNodes.Add(CEBNode);
  }

  for (ExplodedNodeSet::iterator I = CleanedNodes.begin(),
                                 E = CleanedNodes.end(); I != E; ++I) {

    // Step 4: Generate the CallExit and leave the callee's context.
    // CleanedNodes -> CEENode
    CallExitEnd Loc(CE, callerCtx);
    bool isNew;
    ExplodedNode *CEENode = G.getNode(Loc, (*I)->getState(), false, &isNew);
    CEENode->addPredecessor(*I, G);
    if (!isNew)
      return;

    // Step 5: Perform the post-condition check of the CallExpr and enqueue the
    // result onto the work list.
    // CEENode -> Dst -> WorkList
    ExplodedNodeSet Dst;
    NodeBuilderContext Ctx(Engine, calleeCtx->getCallSiteBlock(), CEENode);
    SaveAndRestore<const NodeBuilderContext*> NBCSave(currentBuilderContext,
        &Ctx);
    SaveAndRestore<unsigned> CBISave(currentStmtIdx, calleeCtx->getIndex());

    getCheckerManager().runCheckersForPostStmt(Dst, CEENode, CE, *this, true);

    // Enqueue the next element in the block.
    for (ExplodedNodeSet::iterator PSI = Dst.begin(), PSE = Dst.end();
                                   PSI != PSE; ++PSI) {
      Engine.getWorkList()->enqueue(*PSI, calleeCtx->getCallSiteBlock(),
                                    calleeCtx->getIndex()+1);
    }
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
bool ExprEngine::shouldInlineDecl(const Decl *D, ExplodedNode *Pred) {
  // FIXME: default constructors don't have bodies.
  if (!D->hasBody())
    return false;

  AnalysisDeclContext *CalleeADC = AMgr.getAnalysisDeclContext(D);
  const CFG *CalleeCFG = CalleeADC->getCFG();

  // It is possible that the CFG cannot be constructed.
  // Be safe, and check if the CalleeCFG is valid.
  if (!CalleeCFG)
    return false;

  if (getNumberStackFrames(Pred->getLocationContext())
        == AMgr.InlineMaxStackDepth)
    return false;

  if (Engine.FunctionSummaries->hasReachedMaxBlockCount(D))
    return false;

  if (CalleeCFG->getNumBlockIDs() > AMgr.InlineMaxFunctionSize)
    return false;

  // Do not inline variadic calls (for now).
  if (const BlockDecl *BD = dyn_cast<BlockDecl>(D)) {
    if (BD->isVariadic())
      return false;
  }
  else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->isVariadic())
      return false;
  }

  // Do not inline constructors until we can model destructors.
  // This is unfortunate, but basically necessary for smart pointers and such.
  if (isa<CXXConstructorDecl>(D))
    return false;

  // It is possible that the live variables analysis cannot be
  // run.  If so, bail out.
  if (!CalleeADC->getAnalysis<RelaxedLiveVariables>())
    return false;

  return true;
}

bool ExprEngine::inlineCall(ExplodedNodeSet &Dst,
                            const CallEvent &Call,
                            ExplodedNode *Pred) {
  if (!getAnalysisManager().shouldInlineCall())
    return false;

  const StackFrameContext *CallerSFC =
    Pred->getLocationContext()->getCurrentStackFrame();

  const Decl *D = Call.getDecl();
  const LocationContext *ParentOfCallee = 0;

  switch (Call.getKind()) {
  case CE_Function:
  case CE_CXXConstructor:
  case CE_CXXMember:
    // These are always at least possible to inline.
    break;
  case CE_Block: {
    const BlockDataRegion *BR = cast<BlockCall>(Call).getBlockRegion();
    if (!BR)
      return false;
    D = BR->getDecl();
    AnalysisDeclContext *BlockCtx = AMgr.getAnalysisDeclContext(D);
    ParentOfCallee = BlockCtx->getBlockInvocationContext(CallerSFC,
                                                         cast<BlockDecl>(D),
                                                         BR);
    break;
  }
  case CE_ObjCMessage:
  case CE_ObjCPropertyAccess:
    // These always use dynamic dispatch; enabling inlining means assuming
    // that a particular method will be called at runtime.
    return false;
  }
  
  if (!D || !shouldInlineDecl(D, Pred))
    return false;
  
  if (!ParentOfCallee)
    ParentOfCallee = CallerSFC;

  const Expr *CallE = Call.getOriginExpr();
  assert(CallE && "It is not yet possible to have calls without statements");

  // Construct a new stack frame for the callee.
  AnalysisDeclContext *CalleeADC = AMgr.getAnalysisDeclContext(D);
  const StackFrameContext *CalleeSFC =
    CalleeADC->getStackFrame(ParentOfCallee, CallE,
                             currentBuilderContext->getBlock(),
                             currentStmtIdx);
  
  CallEnter Loc(CallE, CalleeSFC, Pred->getLocationContext());
  bool isNew;
  if (ExplodedNode *N = G.getNode(Loc, Pred->getState(), false, &isNew)) {
    N->addPredecessor(Pred, G);
    if (isNew)
      Engine.getWorkList()->enqueue(N);
  }
  return true;
}

static ProgramStateRef getInlineFailedState(ExplodedNode *&N,
                                            const Stmt *CallE) {
  void *ReplayState = N->getState()->get<ReplayWithoutInlining>();
  if (!ReplayState)
    return 0;
  const Stmt *ReplayCallE = reinterpret_cast<const Stmt *>(ReplayState);
  if (CallE == ReplayCallE) {
    return N->getState()->remove<ReplayWithoutInlining>();
  }
  return 0;
}

void ExprEngine::VisitCallExpr(const CallExpr *CE, ExplodedNode *Pred,
                               ExplodedNodeSet &dst) {
  // Perform the previsit of the CallExpr.
  ExplodedNodeSet dstPreVisit;
  getCheckerManager().runCheckersForPreStmt(dstPreVisit, Pred, CE, *this);

  // Get the callee kind.
  const CXXMemberCallExpr *MemberCE = dyn_cast<CXXMemberCallExpr>(CE);
  bool IsBlock = (MemberCE ? false
                           : CE->getCallee()->getType()->isBlockPointerType());

  // Evaluate the function call.  We try each of the checkers
  // to see if the can evaluate the function call.
  ExplodedNodeSet dstCallEvaluated;
  for (ExplodedNodeSet::iterator I = dstPreVisit.begin(), E = dstPreVisit.end();
       I != E; ++I) {
    ProgramStateRef State = (*I)->getState();
    const LocationContext *LCtx = (*I)->getLocationContext();

    // Evaluate the call.
    if (MemberCE)
      evalCall(dstCallEvaluated, *I, CXXMemberCall(MemberCE, State, LCtx));
    else if (IsBlock)
      evalCall(dstCallEvaluated, *I, BlockCall(CE, State, LCtx));
    else
      evalCall(dstCallEvaluated, *I, FunctionCall(CE, State, LCtx));
  }

  // Finally, perform the post-condition check of the CallExpr and store
  // the created nodes in 'Dst'.
  // Note that if the call was inlined, dstCallEvaluated will be empty.
  // The post-CallExpr check will occur in processCallExit.
  getCheckerManager().runCheckersForPostStmt(dst, dstCallEvaluated, CE,
                                             *this);
}

void ExprEngine::evalCall(ExplodedNodeSet &Dst, ExplodedNode *Pred,
                          const SimpleCall &Call) {
  // Run any pre-call checks using the generic call interface.
  ExplodedNodeSet dstPreVisit;
  getCheckerManager().runCheckersForPreCall(dstPreVisit, Pred, Call, *this);

  // Actually evaluate the function call.  We try each of the checkers
  // to see if the can evaluate the function call, and get a callback at
  // defaultEvalCall if all of them fail.
  ExplodedNodeSet dstCallEvaluated;
  getCheckerManager().runCheckersForEvalCall(dstCallEvaluated, dstPreVisit,
                                             Call, *this);

  // Finally, run any post-call checks.
  getCheckerManager().runCheckersForPostCall(Dst, dstCallEvaluated,
                                             Call, *this);
}

void ExprEngine::defaultEvalCall(ExplodedNodeSet &Dst, ExplodedNode *Pred,
                                 const CallEvent &Call) {
  // Try to inline the call.
  ProgramStateRef state = 0;
  const Expr *E = Call.getOriginExpr();
  if (E) {
    state = getInlineFailedState(Pred, E);
    if (state == 0 && inlineCall(Dst, Call, Pred))
      return;
  }

  // If we can't inline it, handle the return value and invalidate the regions.
  StmtNodeBuilder Bldr(Pred, Dst, *currentBuilderContext);

  // Invalidate any regions touched by the call.
  unsigned Count = currentBuilderContext->getCurrentBlockCount();
  if (state == 0)
    state = Pred->getState();
  state = Call.invalidateRegions(Count, state);

  // Conjure a symbol value to use as the result.
  assert(Call.getOriginExpr() && "Must have an expression to bind the result");
  QualType ResultTy = Call.getResultType();
  SValBuilder &SVB = getSValBuilder();
  const LocationContext *LCtx = Pred->getLocationContext();
  SVal RetVal = SVB.getConjuredSymbolVal(0, Call.getOriginExpr(), LCtx,
                                         ResultTy, Count);

  // And make the result node.
  state = state->BindExpr(Call.getOriginExpr(), LCtx, RetVal);
  Bldr.generateNode(Call.getOriginExpr(), Pred, state);
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
