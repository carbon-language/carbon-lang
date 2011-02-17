//==- CoreEngine.cpp - Path-Sensitive Dataflow Engine ------------*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a generic engine for intraprocedural, path-sensitive,
//  dataflow analysis via graph reachability engine.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CoreEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/Index/TranslationUnit.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/DenseMap.h"
#include <vector>
#include <queue>

using llvm::cast;
using llvm::isa;
using namespace clang;
using namespace ento;

// This should be removed in the future.
namespace clang {
namespace ento {
TransferFuncs* MakeCFRefCountTF(ASTContext& Ctx, bool GCEnabled,
                                  const LangOptions& lopts);
}
}

//===----------------------------------------------------------------------===//
// Worklist classes for exploration of reachable states.
//===----------------------------------------------------------------------===//

WorkList::Visitor::~Visitor() {}

namespace {
class DFS : public WorkList {
  llvm::SmallVector<WorkListUnit,20> Stack;
public:
  virtual bool hasWork() const {
    return !Stack.empty();
  }

  virtual void enqueue(const WorkListUnit& U) {
    Stack.push_back(U);
  }

  virtual WorkListUnit dequeue() {
    assert (!Stack.empty());
    const WorkListUnit& U = Stack.back();
    Stack.pop_back(); // This technically "invalidates" U, but we are fine.
    return U;
  }
  
  virtual bool visitItemsInWorkList(Visitor &V) {
    for (llvm::SmallVectorImpl<WorkListUnit>::iterator
         I = Stack.begin(), E = Stack.end(); I != E; ++I) {
      if (V.visit(*I))
        return true;
    }
    return false;
  }
};

class BFS : public WorkList {
  std::deque<WorkListUnit> Queue;
public:
  virtual bool hasWork() const {
    return !Queue.empty();
  }

  virtual void enqueue(const WorkListUnit& U) {
    Queue.push_front(U);
  }

  virtual WorkListUnit dequeue() {
    WorkListUnit U = Queue.front();
    Queue.pop_front();
    return U;
  }
  
  virtual bool visitItemsInWorkList(Visitor &V) {
    for (std::deque<WorkListUnit>::iterator
         I = Queue.begin(), E = Queue.end(); I != E; ++I) {
      if (V.visit(*I))
        return true;
    }
    return false;
  }
};

} // end anonymous namespace

// Place the dstor for WorkList here because it contains virtual member
// functions, and we the code for the dstor generated in one compilation unit.
WorkList::~WorkList() {}

WorkList *WorkList::makeDFS() { return new DFS(); }
WorkList *WorkList::makeBFS() { return new BFS(); }

namespace {
  class BFSBlockDFSContents : public WorkList {
    std::deque<WorkListUnit> Queue;
    llvm::SmallVector<WorkListUnit,20> Stack;
  public:
    virtual bool hasWork() const {
      return !Queue.empty() || !Stack.empty();
    }

    virtual void enqueue(const WorkListUnit& U) {
      if (isa<BlockEntrance>(U.getNode()->getLocation()))
        Queue.push_front(U);
      else
        Stack.push_back(U);
    }

    virtual WorkListUnit dequeue() {
      // Process all basic blocks to completion.
      if (!Stack.empty()) {
        const WorkListUnit& U = Stack.back();
        Stack.pop_back(); // This technically "invalidates" U, but we are fine.
        return U;
      }

      assert(!Queue.empty());
      // Don't use const reference.  The subsequent pop_back() might make it
      // unsafe.
      WorkListUnit U = Queue.front();
      Queue.pop_front();
      return U;
    }
    virtual bool visitItemsInWorkList(Visitor &V) {
      for (llvm::SmallVectorImpl<WorkListUnit>::iterator
           I = Stack.begin(), E = Stack.end(); I != E; ++I) {
        if (V.visit(*I))
          return true;
      }
      for (std::deque<WorkListUnit>::iterator
           I = Queue.begin(), E = Queue.end(); I != E; ++I) {
        if (V.visit(*I))
          return true;
      }
      return false;
    }

  };
} // end anonymous namespace

WorkList* WorkList::makeBFSBlockDFSContents() {
  return new BFSBlockDFSContents();
}

//===----------------------------------------------------------------------===//
// Core analysis engine.
//===----------------------------------------------------------------------===//

/// ExecuteWorkList - Run the worklist algorithm for a maximum number of steps.
bool CoreEngine::ExecuteWorkList(const LocationContext *L, unsigned Steps,
                                   const GRState *InitState) {

  if (G->num_roots() == 0) { // Initialize the analysis by constructing
    // the root if none exists.

    const CFGBlock* Entry = &(L->getCFG()->getEntry());

    assert (Entry->empty() &&
            "Entry block must be empty.");

    assert (Entry->succ_size() == 1 &&
            "Entry block must have 1 successor.");

    // Get the solitary successor.
    const CFGBlock* Succ = *(Entry->succ_begin());

    // Construct an edge representing the
    // starting location in the function.
    BlockEdge StartLoc(Entry, Succ, L);

    // Set the current block counter to being empty.
    WList->setBlockCounter(BCounterFactory.GetEmptyCounter());

    if (!InitState)
      // Generate the root.
      generateNode(StartLoc, SubEng.getInitialState(L), 0);
    else
      generateNode(StartLoc, InitState, 0);
  }

  // Check if we have a steps limit
  bool UnlimitedSteps = Steps == 0;

  while (WList->hasWork()) {
    if (!UnlimitedSteps) {
      if (Steps == 0)
        break;
      --Steps;
    }

    const WorkListUnit& WU = WList->dequeue();

    // Set the current block counter.
    WList->setBlockCounter(WU.getBlockCounter());

    // Retrieve the node.
    ExplodedNode* Node = WU.getNode();

    // Dispatch on the location type.
    switch (Node->getLocation().getKind()) {
      case ProgramPoint::BlockEdgeKind:
        HandleBlockEdge(cast<BlockEdge>(Node->getLocation()), Node);
        break;

      case ProgramPoint::BlockEntranceKind:
        HandleBlockEntrance(cast<BlockEntrance>(Node->getLocation()), Node);
        break;

      case ProgramPoint::BlockExitKind:
        assert (false && "BlockExit location never occur in forward analysis.");
        break;

      case ProgramPoint::CallEnterKind:
        HandleCallEnter(cast<CallEnter>(Node->getLocation()), WU.getBlock(), 
                        WU.getIndex(), Node);
        break;

      case ProgramPoint::CallExitKind:
        HandleCallExit(cast<CallExit>(Node->getLocation()), Node);
        break;

      default:
        assert(isa<PostStmt>(Node->getLocation()) || 
               isa<PostInitializer>(Node->getLocation()));
        HandlePostStmt(WU.getBlock(), WU.getIndex(), Node);
        break;
    }
  }

  SubEng.processEndWorklist(hasWorkRemaining());
  return WList->hasWork();
}

void CoreEngine::ExecuteWorkListWithInitialState(const LocationContext *L, 
                                                   unsigned Steps,
                                                   const GRState *InitState, 
                                                   ExplodedNodeSet &Dst) {
  ExecuteWorkList(L, Steps, InitState);
  for (llvm::SmallVectorImpl<ExplodedNode*>::iterator I = G->EndNodes.begin(), 
                                           E = G->EndNodes.end(); I != E; ++I) {
    Dst.Add(*I);
  }
}

void CoreEngine::HandleCallEnter(const CallEnter &L, const CFGBlock *Block,
                                   unsigned Index, ExplodedNode *Pred) {
  CallEnterNodeBuilder Builder(*this, Pred, L.getCallExpr(), 
                                 L.getCalleeContext(), Block, Index);
  SubEng.processCallEnter(Builder);
}

void CoreEngine::HandleCallExit(const CallExit &L, ExplodedNode *Pred) {
  CallExitNodeBuilder Builder(*this, Pred);
  SubEng.processCallExit(Builder);
}

void CoreEngine::HandleBlockEdge(const BlockEdge& L, ExplodedNode* Pred) {

  const CFGBlock* Blk = L.getDst();

  // Check if we are entering the EXIT block.
  if (Blk == &(L.getLocationContext()->getCFG()->getExit())) {

    assert (L.getLocationContext()->getCFG()->getExit().size() == 0
            && "EXIT block cannot contain Stmts.");

    // Process the final state transition.
    EndOfFunctionNodeBuilder Builder(Blk, Pred, this);
    SubEng.processEndOfFunction(Builder);

    // This path is done. Don't enqueue any more nodes.
    return;
  }

  // Call into the subengine to process entering the CFGBlock.
  ExplodedNodeSet dstNodes;
  BlockEntrance BE(Blk, Pred->getLocationContext());
  GenericNodeBuilder<BlockEntrance> nodeBuilder(*this, Pred, BE);
  SubEng.processCFGBlockEntrance(dstNodes, nodeBuilder);

  if (dstNodes.empty()) {
    if (!nodeBuilder.hasGeneratedNode) {
      // Auto-generate a node and enqueue it to the worklist.
      generateNode(BE, Pred->State, Pred);    
    }
  }
  else {
    for (ExplodedNodeSet::iterator I = dstNodes.begin(), E = dstNodes.end();
         I != E; ++I) {
      WList->enqueue(*I);
    }
  }

  for (llvm::SmallVectorImpl<ExplodedNode*>::const_iterator
       I = nodeBuilder.sinks().begin(), E = nodeBuilder.sinks().end();
       I != E; ++I) {
    blocksAborted.push_back(std::make_pair(L, *I));
  }
}

void CoreEngine::HandleBlockEntrance(const BlockEntrance& L,
                                       ExplodedNode* Pred) {

  // Increment the block counter.
  BlockCounter Counter = WList->getBlockCounter();
  Counter = BCounterFactory.IncrementCount(Counter, 
                             Pred->getLocationContext()->getCurrentStackFrame(),
                                           L.getBlock()->getBlockID());
  WList->setBlockCounter(Counter);

  // Process the entrance of the block.
  if (CFGElement E = L.getFirstElement()) {
    StmtNodeBuilder Builder(L.getBlock(), 0, Pred, this,
                              SubEng.getStateManager());
    SubEng.processCFGElement(E, Builder);
  }
  else
    HandleBlockExit(L.getBlock(), Pred);
}

void CoreEngine::HandleBlockExit(const CFGBlock * B, ExplodedNode* Pred) {

  if (const Stmt* Term = B->getTerminator()) {
    switch (Term->getStmtClass()) {
      default:
        assert(false && "Analysis for this terminator not implemented.");
        break;

      case Stmt::BinaryOperatorClass: // '&&' and '||'
        HandleBranch(cast<BinaryOperator>(Term)->getLHS(), Term, B, Pred);
        return;

      case Stmt::BinaryConditionalOperatorClass:
      case Stmt::ConditionalOperatorClass:
        HandleBranch(cast<AbstractConditionalOperator>(Term)->getCond(),
                     Term, B, Pred);
        return;

        // FIXME: Use constant-folding in CFG construction to simplify this
        // case.

      case Stmt::ChooseExprClass:
        HandleBranch(cast<ChooseExpr>(Term)->getCond(), Term, B, Pred);
        return;

      case Stmt::DoStmtClass:
        HandleBranch(cast<DoStmt>(Term)->getCond(), Term, B, Pred);
        return;

      case Stmt::ForStmtClass:
        HandleBranch(cast<ForStmt>(Term)->getCond(), Term, B, Pred);
        return;

      case Stmt::ContinueStmtClass:
      case Stmt::BreakStmtClass:
      case Stmt::GotoStmtClass:
        break;

      case Stmt::IfStmtClass:
        HandleBranch(cast<IfStmt>(Term)->getCond(), Term, B, Pred);
        return;

      case Stmt::IndirectGotoStmtClass: {
        // Only 1 successor: the indirect goto dispatch block.
        assert (B->succ_size() == 1);

        IndirectGotoNodeBuilder
           builder(Pred, B, cast<IndirectGotoStmt>(Term)->getTarget(),
                   *(B->succ_begin()), this);

        SubEng.processIndirectGoto(builder);
        return;
      }

      case Stmt::ObjCForCollectionStmtClass: {
        // In the case of ObjCForCollectionStmt, it appears twice in a CFG:
        //
        //  (1) inside a basic block, which represents the binding of the
        //      'element' variable to a value.
        //  (2) in a terminator, which represents the branch.
        //
        // For (1), subengines will bind a value (i.e., 0 or 1) indicating
        // whether or not collection contains any more elements.  We cannot
        // just test to see if the element is nil because a container can
        // contain nil elements.
        HandleBranch(Term, Term, B, Pred);
        return;
      }

      case Stmt::SwitchStmtClass: {
        SwitchNodeBuilder builder(Pred, B, cast<SwitchStmt>(Term)->getCond(),
                                    this);

        SubEng.processSwitch(builder);
        return;
      }

      case Stmt::WhileStmtClass:
        HandleBranch(cast<WhileStmt>(Term)->getCond(), Term, B, Pred);
        return;
    }
  }

  assert (B->succ_size() == 1 &&
          "Blocks with no terminator should have at most 1 successor.");

  generateNode(BlockEdge(B, *(B->succ_begin()), Pred->getLocationContext()),
               Pred->State, Pred);
}

void CoreEngine::HandleBranch(const Stmt* Cond, const Stmt* Term, 
                                const CFGBlock * B, ExplodedNode* Pred) {
  assert(B->succ_size() == 2);
  BranchNodeBuilder Builder(B, *(B->succ_begin()), *(B->succ_begin()+1),
                            Pred, this);
  SubEng.processBranch(Cond, Term, Builder);
}

void CoreEngine::HandlePostStmt(const CFGBlock* B, unsigned StmtIdx, 
                                  ExplodedNode* Pred) {
  assert (!B->empty());

  if (StmtIdx == B->size())
    HandleBlockExit(B, Pred);
  else {
    StmtNodeBuilder Builder(B, StmtIdx, Pred, this,
                              SubEng.getStateManager());
    SubEng.processCFGElement((*B)[StmtIdx], Builder);
  }
}

/// generateNode - Utility method to generate nodes, hook up successors,
///  and add nodes to the worklist.
void CoreEngine::generateNode(const ProgramPoint& Loc,
                              const GRState* State, ExplodedNode* Pred) {

  bool IsNew;
  ExplodedNode* Node = G->getNode(Loc, State, &IsNew);

  if (Pred)
    Node->addPredecessor(Pred, *G);  // Link 'Node' with its predecessor.
  else {
    assert (IsNew);
    G->addRoot(Node);  // 'Node' has no predecessor.  Make it a root.
  }

  // Only add 'Node' to the worklist if it was freshly generated.
  if (IsNew) WList->enqueue(Node);
}

ExplodedNode *
GenericNodeBuilderImpl::generateNodeImpl(const GRState *state,
                                         ExplodedNode *pred,
                                         ProgramPoint programPoint,
                                         bool asSink) {
  
  hasGeneratedNode = true;
  bool isNew;
  ExplodedNode *node = engine.getGraph().getNode(programPoint, state, &isNew);
  if (pred)
    node->addPredecessor(pred, engine.getGraph());
  if (isNew) {
    if (asSink) {
      node->markAsSink();
      sinksGenerated.push_back(node);
    }
    return node;
  }
  return 0;
}

StmtNodeBuilder::StmtNodeBuilder(const CFGBlock* b, unsigned idx,
                                     ExplodedNode* N, CoreEngine* e,
                                     GRStateManager &mgr)
  : Eng(*e), B(*b), Idx(idx), Pred(N), Mgr(mgr),
    PurgingDeadSymbols(false), BuildSinks(false), hasGeneratedNode(false),
    PointKind(ProgramPoint::PostStmtKind), Tag(0) {
  Deferred.insert(N);
  CleanedState = Pred->getState();
}

StmtNodeBuilder::~StmtNodeBuilder() {
  for (DeferredTy::iterator I=Deferred.begin(), E=Deferred.end(); I!=E; ++I)
    if (!(*I)->isSink())
      GenerateAutoTransition(*I);
}

void StmtNodeBuilder::GenerateAutoTransition(ExplodedNode* N) {
  assert (!N->isSink());

  // Check if this node entered a callee.
  if (isa<CallEnter>(N->getLocation())) {
    // Still use the index of the CallExpr. It's needed to create the callee
    // StackFrameContext.
    Eng.WList->enqueue(N, &B, Idx);
    return;
  }

  // Do not create extra nodes. Move to the next CFG element.
  if (isa<PostInitializer>(N->getLocation())) {
    Eng.WList->enqueue(N, &B, Idx+1);
    return;
  }

  PostStmt Loc(getStmt(), N->getLocationContext());

  if (Loc == N->getLocation()) {
    // Note: 'N' should be a fresh node because otherwise it shouldn't be
    // a member of Deferred.
    Eng.WList->enqueue(N, &B, Idx+1);
    return;
  }

  bool IsNew;
  ExplodedNode* Succ = Eng.G->getNode(Loc, N->State, &IsNew);
  Succ->addPredecessor(N, *Eng.G);

  if (IsNew)
    Eng.WList->enqueue(Succ, &B, Idx+1);
}

ExplodedNode* StmtNodeBuilder::MakeNode(ExplodedNodeSet& Dst, const Stmt* S, 
                                          ExplodedNode* Pred, const GRState* St,
                                          ProgramPoint::Kind K) {

  ExplodedNode* N = generateNode(S, St, Pred, K);

  if (N) {
    if (BuildSinks)
      N->markAsSink();
    else
      Dst.Add(N);
  }
  
  return N;
}

static ProgramPoint GetProgramPoint(const Stmt *S, ProgramPoint::Kind K,
                                    const LocationContext *LC, const void *tag){
  switch (K) {
    default:
      assert(false && "Unhandled ProgramPoint kind");    
    case ProgramPoint::PreStmtKind:
      return PreStmt(S, LC, tag);
    case ProgramPoint::PostStmtKind:
      return PostStmt(S, LC, tag);
    case ProgramPoint::PreLoadKind:
      return PreLoad(S, LC, tag);
    case ProgramPoint::PostLoadKind:
      return PostLoad(S, LC, tag);
    case ProgramPoint::PreStoreKind:
      return PreStore(S, LC, tag);
    case ProgramPoint::PostStoreKind:
      return PostStore(S, LC, tag);
    case ProgramPoint::PostLValueKind:
      return PostLValue(S, LC, tag);
    case ProgramPoint::PostPurgeDeadSymbolsKind:
      return PostPurgeDeadSymbols(S, LC, tag);
  }
}

ExplodedNode*
StmtNodeBuilder::generateNodeInternal(const Stmt* S, const GRState* state,
                                        ExplodedNode* Pred,
                                        ProgramPoint::Kind K,
                                        const void *tag) {
  
  const ProgramPoint &L = GetProgramPoint(S, K, Pred->getLocationContext(),tag);
  return generateNodeInternal(L, state, Pred);
}

ExplodedNode*
StmtNodeBuilder::generateNodeInternal(const ProgramPoint &Loc,
                                        const GRState* State,
                                        ExplodedNode* Pred) {
  bool IsNew;
  ExplodedNode* N = Eng.G->getNode(Loc, State, &IsNew);
  N->addPredecessor(Pred, *Eng.G);
  Deferred.erase(Pred);

  if (IsNew) {
    Deferred.insert(N);
    return N;
  }

  return NULL;
}

ExplodedNode* BranchNodeBuilder::generateNode(const GRState* State,
                                                bool branch) {

  // If the branch has been marked infeasible we should not generate a node.
  if (!isFeasible(branch))
    return NULL;

  bool IsNew;

  ExplodedNode* Succ =
    Eng.G->getNode(BlockEdge(Src,branch ? DstT:DstF,Pred->getLocationContext()),
                   State, &IsNew);

  Succ->addPredecessor(Pred, *Eng.G);

  if (branch)
    GeneratedTrue = true;
  else
    GeneratedFalse = true;

  if (IsNew) {
    Deferred.push_back(Succ);
    return Succ;
  }

  return NULL;
}

BranchNodeBuilder::~BranchNodeBuilder() {
  if (!GeneratedTrue) generateNode(Pred->State, true);
  if (!GeneratedFalse) generateNode(Pred->State, false);

  for (DeferredTy::iterator I=Deferred.begin(), E=Deferred.end(); I!=E; ++I)
    if (!(*I)->isSink()) Eng.WList->enqueue(*I);
}


ExplodedNode*
IndirectGotoNodeBuilder::generateNode(const iterator& I, const GRState* St,
                                        bool isSink) {
  bool IsNew;

  ExplodedNode* Succ = Eng.G->getNode(BlockEdge(Src, I.getBlock(),
                                      Pred->getLocationContext()), St, &IsNew);

  Succ->addPredecessor(Pred, *Eng.G);

  if (IsNew) {

    if (isSink)
      Succ->markAsSink();
    else
      Eng.WList->enqueue(Succ);

    return Succ;
  }

  return NULL;
}


ExplodedNode*
SwitchNodeBuilder::generateCaseStmtNode(const iterator& I, const GRState* St){

  bool IsNew;

  ExplodedNode* Succ = Eng.G->getNode(BlockEdge(Src, I.getBlock(),
                                       Pred->getLocationContext()), St, &IsNew);
  Succ->addPredecessor(Pred, *Eng.G);

  if (IsNew) {
    Eng.WList->enqueue(Succ);
    return Succ;
  }

  return NULL;
}


ExplodedNode*
SwitchNodeBuilder::generateDefaultCaseNode(const GRState* St, bool isSink) {

  // Get the block for the default case.
  assert (Src->succ_rbegin() != Src->succ_rend());
  CFGBlock* DefaultBlock = *Src->succ_rbegin();

  bool IsNew;

  ExplodedNode* Succ = Eng.G->getNode(BlockEdge(Src, DefaultBlock,
                                       Pred->getLocationContext()), St, &IsNew);
  Succ->addPredecessor(Pred, *Eng.G);

  if (IsNew) {
    if (isSink)
      Succ->markAsSink();
    else
      Eng.WList->enqueue(Succ);

    return Succ;
  }

  return NULL;
}

EndOfFunctionNodeBuilder::~EndOfFunctionNodeBuilder() {
  // Auto-generate an EOP node if one has not been generated.
  if (!hasGeneratedNode) {
    // If we are in an inlined call, generate CallExit node.
    if (Pred->getLocationContext()->getParent())
      GenerateCallExitNode(Pred->State);
    else
      generateNode(Pred->State);
  }
}

ExplodedNode*
EndOfFunctionNodeBuilder::generateNode(const GRState* State, const void *tag,
                                   ExplodedNode* P) {
  hasGeneratedNode = true;
  bool IsNew;

  ExplodedNode* Node = Eng.G->getNode(BlockEntrance(&B,
                               Pred->getLocationContext(), tag), State, &IsNew);

  Node->addPredecessor(P ? P : Pred, *Eng.G);

  if (IsNew) {
    Eng.G->addEndOfPath(Node);
    return Node;
  }

  return NULL;
}

void EndOfFunctionNodeBuilder::GenerateCallExitNode(const GRState *state) {
  hasGeneratedNode = true;
  // Create a CallExit node and enqueue it.
  const StackFrameContext *LocCtx
                         = cast<StackFrameContext>(Pred->getLocationContext());
  const Stmt *CE = LocCtx->getCallSite();

  // Use the the callee location context.
  CallExit Loc(CE, LocCtx);

  bool isNew;
  ExplodedNode *Node = Eng.G->getNode(Loc, state, &isNew);
  Node->addPredecessor(Pred, *Eng.G);

  if (isNew)
    Eng.WList->enqueue(Node);
}
                                                

void CallEnterNodeBuilder::generateNode(const GRState *state) {
  // Check if the callee is in the same translation unit.
  if (CalleeCtx->getTranslationUnit() != 
      Pred->getLocationContext()->getTranslationUnit()) {
    // Create a new engine. We must be careful that the new engine should not
    // reference data structures owned by the old engine.

    AnalysisManager &OldMgr = Eng.SubEng.getAnalysisManager();
    
    // Get the callee's translation unit.
    idx::TranslationUnit *TU = CalleeCtx->getTranslationUnit();

    // Create a new AnalysisManager with components of the callee's
    // TranslationUnit.
    // The Diagnostic is actually shared when we create ASTUnits from AST files.
    AnalysisManager AMgr(TU->getASTContext(), TU->getDiagnostic(), 
                         OldMgr.getLangOptions(), 
                         OldMgr.getPathDiagnosticClient(),
                         OldMgr.getStoreManagerCreator(),
                         OldMgr.getConstraintManagerCreator(),
                         OldMgr.getCheckerManager(),
                         OldMgr.getIndexer(),
                         OldMgr.getMaxNodes(), OldMgr.getMaxVisit(),
                         OldMgr.shouldVisualizeGraphviz(),
                         OldMgr.shouldVisualizeUbigraph(),
                         OldMgr.shouldPurgeDead(),
                         OldMgr.shouldEagerlyAssume(),
                         OldMgr.shouldTrimGraph(),
                         OldMgr.shouldInlineCall(),
                     OldMgr.getAnalysisContextManager().getUseUnoptimizedCFG(),
                     OldMgr.getAnalysisContextManager().getAddImplicitDtors(),
                     OldMgr.getAnalysisContextManager().getAddInitializers(),
                     OldMgr.shouldEagerlyTrimExplodedGraph());
    llvm::OwningPtr<TransferFuncs> TF(MakeCFRefCountTF(AMgr.getASTContext(),
                                                         /* GCEnabled */ false,
                                                        AMgr.getLangOptions()));
    // Create the new engine.
    ExprEngine NewEng(AMgr, TF.take());

    // Create the new LocationContext.
    AnalysisContext *NewAnaCtx = AMgr.getAnalysisContext(CalleeCtx->getDecl(), 
                                               CalleeCtx->getTranslationUnit());
    const StackFrameContext *OldLocCtx = CalleeCtx;
    const StackFrameContext *NewLocCtx = AMgr.getStackFrame(NewAnaCtx, 
                                               OldLocCtx->getParent(),
                                               OldLocCtx->getCallSite(),
                                               OldLocCtx->getCallSiteBlock(), 
                                               OldLocCtx->getIndex());

    // Now create an initial state for the new engine.
    const GRState *NewState = NewEng.getStateManager().MarshalState(state,
                                                                    NewLocCtx);
    ExplodedNodeSet ReturnNodes;
    NewEng.ExecuteWorkListWithInitialState(NewLocCtx, AMgr.getMaxNodes(), 
                                           NewState, ReturnNodes);
    return;
  }

  // Get the callee entry block.
  const CFGBlock *Entry = &(CalleeCtx->getCFG()->getEntry());
  assert(Entry->empty());
  assert(Entry->succ_size() == 1);

  // Get the solitary successor.
  const CFGBlock *SuccB = *(Entry->succ_begin());

  // Construct an edge representing the starting location in the callee.
  BlockEdge Loc(Entry, SuccB, CalleeCtx);

  bool isNew;
  ExplodedNode *Node = Eng.G->getNode(Loc, state, &isNew);
  Node->addPredecessor(const_cast<ExplodedNode*>(Pred), *Eng.G);

  if (isNew)
    Eng.WList->enqueue(Node);
}

void CallExitNodeBuilder::generateNode(const GRState *state) {
  // Get the callee's location context.
  const StackFrameContext *LocCtx 
                         = cast<StackFrameContext>(Pred->getLocationContext());
  // When exiting an implicit automatic obj dtor call, the callsite is the Stmt
  // that triggers the dtor.
  PostStmt Loc(LocCtx->getCallSite(), LocCtx->getParent());
  bool isNew;
  ExplodedNode *Node = Eng.G->getNode(Loc, state, &isNew);
  Node->addPredecessor(const_cast<ExplodedNode*>(Pred), *Eng.G);
  if (isNew)
    Eng.WList->enqueue(Node, LocCtx->getCallSiteBlock(),
                       LocCtx->getIndex() + 1);
}
