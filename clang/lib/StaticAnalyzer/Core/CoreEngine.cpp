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
#include "clang/AST/StmtCXX.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/DenseMap.h"
using namespace clang;
using namespace ento;

//===----------------------------------------------------------------------===//
// Worklist classes for exploration of reachable states.
//===----------------------------------------------------------------------===//

WorkList::Visitor::~Visitor() {}

namespace {
class DFS : public WorkList {
  SmallVector<WorkListUnit,20> Stack;
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
    for (SmallVectorImpl<WorkListUnit>::iterator
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
    SmallVector<WorkListUnit,20> Stack;
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
      for (SmallVectorImpl<WorkListUnit>::iterator
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
                                   const ProgramState *InitState) {

  if (G->num_roots() == 0) { // Initialize the analysis by constructing
    // the root if none exists.

    const CFGBlock *Entry = &(L->getCFG()->getEntry());

    assert (Entry->empty() &&
            "Entry block must be empty.");

    assert (Entry->succ_size() == 1 &&
            "Entry block must have 1 successor.");

    // Get the solitary successor.
    const CFGBlock *Succ = *(Entry->succ_begin());

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
    ExplodedNode *Node = WU.getNode();

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
                                                 const ProgramState *InitState, 
                                                 ExplodedNodeSet &Dst) {
  ExecuteWorkList(L, Steps, InitState);
  for (SmallVectorImpl<ExplodedNode*>::iterator I = G->EndNodes.begin(), 
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

void CoreEngine::HandleBlockEdge(const BlockEdge &L, ExplodedNode *Pred) {

  const CFGBlock *Blk = L.getDst();
  NodeBuilderContext BuilderCtx(*this, Blk, Pred);

  // Check if we are entering the EXIT block.
  if (Blk == &(L.getLocationContext()->getCFG()->getExit())) {

    assert (L.getLocationContext()->getCFG()->getExit().size() == 0
            && "EXIT block cannot contain Stmts.");

    // Process the final state transition.
    SubEng.processEndOfFunction(BuilderCtx);

    // This path is done. Don't enqueue any more nodes.
    return;
  }

  // Call into the SubEngine to process entering the CFGBlock.
  ExplodedNodeSet dstNodes;
  BlockEntrance BE(Blk, Pred->getLocationContext());
  NodeBuilderWithSinks nodeBuilder(Pred, dstNodes, BuilderCtx, BE);
  SubEng.processCFGBlockEntrance(nodeBuilder);

  // Auto-generate a node.
  if (!nodeBuilder.hasGeneratedNodes()) {
    nodeBuilder.generateNode(Pred->State, Pred);
  }

  // Enqueue nodes onto the worklist.
  enqueue(dstNodes);

  // Make sink nodes as exhausted.
  const SmallVectorImpl<ExplodedNode*> &Sinks =  nodeBuilder.getSinks();
  for (SmallVectorImpl<ExplodedNode*>::const_iterator
         I =Sinks.begin(), E = Sinks.end(); I != E; ++I) {
    blocksExhausted.push_back(std::make_pair(L, *I));
  }
}

void CoreEngine::HandleBlockEntrance(const BlockEntrance &L,
                                       ExplodedNode *Pred) {

  // Increment the block counter.
  BlockCounter Counter = WList->getBlockCounter();
  Counter = BCounterFactory.IncrementCount(Counter, 
                             Pred->getLocationContext()->getCurrentStackFrame(),
                                           L.getBlock()->getBlockID());
  WList->setBlockCounter(Counter);

  // Process the entrance of the block.
  if (CFGElement E = L.getFirstElement()) {
    NodeBuilderContext Ctx(*this, L.getBlock(), Pred);
    SubEng.processCFGElement(E, Pred, 0, &Ctx);
  }
  else
    HandleBlockExit(L.getBlock(), Pred);
}

void CoreEngine::HandleBlockExit(const CFGBlock * B, ExplodedNode *Pred) {

  if (const Stmt *Term = B->getTerminator()) {
    switch (Term->getStmtClass()) {
      default:
        llvm_unreachable("Analysis for this terminator not implemented.");

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

      case Stmt::CXXForRangeStmtClass:
        HandleBranch(cast<CXXForRangeStmt>(Term)->getCond(), Term, B, Pred);
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

void CoreEngine::HandleBranch(const Stmt *Cond, const Stmt *Term, 
                                const CFGBlock * B, ExplodedNode *Pred) {
  assert(B->succ_size() == 2);
  NodeBuilderContext Ctx(*this, B, Pred);
  ExplodedNodeSet Dst;
  SubEng.processBranch(Cond, Term, Ctx, Pred, Dst,
                       *(B->succ_begin()), *(B->succ_begin()+1));
  // Enqueue the new frontier onto the worklist.
  enqueue(Dst);
}

void CoreEngine::HandlePostStmt(const CFGBlock *B, unsigned StmtIdx, 
                                  ExplodedNode *Pred) {
  assert(B);
  assert(!B->empty());

  if (StmtIdx == B->size())
    HandleBlockExit(B, Pred);
  else {
    NodeBuilderContext Ctx(*this, B, Pred);
    SubEng.processCFGElement((*B)[StmtIdx], Pred, StmtIdx, &Ctx);
  }
}

/// generateNode - Utility method to generate nodes, hook up successors,
///  and add nodes to the worklist.
void CoreEngine::generateNode(const ProgramPoint &Loc,
                              const ProgramState *State,
                              ExplodedNode *Pred) {

  bool IsNew;
  ExplodedNode *Node = G->getNode(Loc, State, &IsNew);

  if (Pred)
    Node->addPredecessor(Pred, *G);  // Link 'Node' with its predecessor.
  else {
    assert (IsNew);
    G->addRoot(Node);  // 'Node' has no predecessor.  Make it a root.
  }

  // Only add 'Node' to the worklist if it was freshly generated.
  if (IsNew) WList->enqueue(Node);
}

void CoreEngine::enqueueStmtNode(ExplodedNode *N,
                                 const CFGBlock *Block, unsigned Idx) {
  assert (!N->isSink());

  // Check if this node entered a callee.
  if (isa<CallEnter>(N->getLocation())) {
    // Still use the index of the CallExpr. It's needed to create the callee
    // StackFrameContext.
    WList->enqueue(N, Block, Idx);
    return;
  }

  // Do not create extra nodes. Move to the next CFG element.
  if (isa<PostInitializer>(N->getLocation())) {
    WList->enqueue(N, Block, Idx+1);
    return;
  }

  const CFGStmt *CS = (*Block)[Idx].getAs<CFGStmt>();
  const Stmt *St = CS ? CS->getStmt() : 0;
  PostStmt Loc(St, N->getLocationContext());

  if (Loc == N->getLocation()) {
    // Note: 'N' should be a fresh node because otherwise it shouldn't be
    // a member of Deferred.
    WList->enqueue(N, Block, Idx+1);
    return;
  }

  bool IsNew;
  ExplodedNode *Succ = G->getNode(Loc, N->getState(), &IsNew);
  Succ->addPredecessor(N, *G);

  if (IsNew)
    WList->enqueue(Succ, Block, Idx+1);
}

void CoreEngine::enqueue(ExplodedNodeSet &Set) {
  for (ExplodedNodeSet::iterator I = Set.begin(),
                                 E = Set.end(); I != E; ++I) {
    WList->enqueue(*I);
  }
}

void CoreEngine::enqueue(ExplodedNodeSet &Set,
                         const CFGBlock *Block, unsigned Idx) {
  for (ExplodedNodeSet::iterator I = Set.begin(),
                                 E = Set.end(); I != E; ++I) {
    enqueueStmtNode(*I, Block, Idx);
  }
}


ExplodedNode* NodeBuilder::generateNodeImpl(const ProgramPoint &Loc,
                                            const ProgramState *State,
                                            ExplodedNode *FromN,
                                            bool MarkAsSink) {
  HasGeneratedNodes = true;
  bool IsNew;
  ExplodedNode *N = C.Eng.G->getNode(Loc, State, &IsNew);
  N->addPredecessor(FromN, *C.Eng.G);
  Frontier.erase(FromN);

  if (MarkAsSink)
    N->markAsSink();
    
  if (IsNew && !MarkAsSink)
    Frontier.Add(N);

  return (IsNew ? N : 0);
}

StmtNodeBuilder::~StmtNodeBuilder() {
  if (EnclosingBldr)
    for (ExplodedNodeSet::iterator I = Frontier.begin(),
                                   E = Frontier.end(); I != E; ++I )
      EnclosingBldr->addNodes(*I);
}

ExplodedNode *BranchNodeBuilder::generateNode(const ProgramState *State,
                                              bool branch,
                                              ExplodedNode *NodePred) {
  // If the branch has been marked infeasible we should not generate a node.
  if (!isFeasible(branch))
    return NULL;

  ProgramPoint Loc = BlockEdge(C.Block, branch ? DstT:DstF,
                               NodePred->getLocationContext());
  ExplodedNode *Succ = generateNodeImpl(Loc, State, NodePred);
  return Succ;
}

ExplodedNode*
IndirectGotoNodeBuilder::generateNode(const iterator &I,
                                      const ProgramState *St,
                                      bool isSink) {
  bool IsNew;

  ExplodedNode *Succ = Eng.G->getNode(BlockEdge(Src, I.getBlock(),
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
SwitchNodeBuilder::generateCaseStmtNode(const iterator &I,
                                        const ProgramState *St) {

  bool IsNew;
  ExplodedNode *Succ = Eng.G->getNode(BlockEdge(Src, I.getBlock(),
                                      Pred->getLocationContext()),
                                      St, &IsNew);
  Succ->addPredecessor(Pred, *Eng.G);
  if (IsNew) {
    Eng.WList->enqueue(Succ);
    return Succ;
  }
  return NULL;
}


ExplodedNode*
SwitchNodeBuilder::generateDefaultCaseNode(const ProgramState *St,
                                           bool isSink) {
  // Get the block for the default case.
  assert(Src->succ_rbegin() != Src->succ_rend());
  CFGBlock *DefaultBlock = *Src->succ_rbegin();

  // Sanity check for default blocks that are unreachable and not caught
  // by earlier stages.
  if (!DefaultBlock)
    return NULL;
  
  bool IsNew;

  ExplodedNode *Succ = Eng.G->getNode(BlockEdge(Src, DefaultBlock,
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

void CallEnterNodeBuilder::generateNode(const ProgramState *state) {
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
    // The Diagnostic is  actually shared when we create ASTUnits from AST files.
    AnalysisManager AMgr(TU->getASTContext(), TU->getDiagnostic(), OldMgr);

    // Create the new engine.
    // FIXME: This cast isn't really safe.
    bool GCEnabled = static_cast<ExprEngine&>(Eng.SubEng).isObjCGCEnabled();
    ExprEngine NewEng(AMgr, GCEnabled);

    // Create the new LocationContext.
    AnalysisDeclContext *NewAnaCtx =
      AMgr.getAnalysisDeclContext(CalleeCtx->getDecl(), 
                              CalleeCtx->getTranslationUnit());

    const StackFrameContext *OldLocCtx = CalleeCtx;
    const StackFrameContext *NewLocCtx =
      NewAnaCtx->getStackFrame(OldLocCtx->getParent(),
                               OldLocCtx->getCallSite(),
                               OldLocCtx->getCallSiteBlock(), 
                               OldLocCtx->getIndex());

    // Now create an initial state for the new engine.
    const ProgramState *NewState =
      NewEng.getStateManager().MarshalState(state, NewLocCtx);
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

void CallExitNodeBuilder::generateNode(const ProgramState *state) {
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
