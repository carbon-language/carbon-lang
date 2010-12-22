//==- CoreEngine.h - Path-Sensitive Dataflow Engine ----------------*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a generic engine for intraprocedural, path-sensitive,
//  dataflow analysis via graph reachability.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_COREENGINE
#define LLVM_CLANG_GR_COREENGINE

#include "clang/AST/Expr.h"
#include "clang/GR/PathSensitive/ExplodedGraph.h"
#include "clang/GR/PathSensitive/WorkList.h"
#include "clang/GR/PathSensitive/BlockCounter.h"
#include "clang/GR/PathSensitive/SubEngine.h"
#include "llvm/ADT/OwningPtr.h"

namespace clang {

namespace GR {

//===----------------------------------------------------------------------===//
/// CoreEngine - Implements the core logic of the graph-reachability
///   analysis. It traverses the CFG and generates the ExplodedGraph.
///   Program "states" are treated as opaque void pointers.
///   The template class CoreEngine (which subclasses CoreEngine)
///   provides the matching component to the engine that knows the actual types
///   for states.  Note that this engine only dispatches to transfer functions
///   at the statement and block-level.  The analyses themselves must implement
///   any transfer function logic and the sub-expression level (if any).
class CoreEngine {
  friend class StmtNodeBuilder;
  friend class BranchNodeBuilder;
  friend class IndirectGotoNodeBuilder;
  friend class SwitchNodeBuilder;
  friend class EndPathNodeBuilder;
  friend class CallEnterNodeBuilder;
  friend class CallExitNodeBuilder;

public:
  typedef std::vector<std::pair<BlockEdge, const ExplodedNode*> >
            BlocksAborted;
private:

  SubEngine& SubEng;

  /// G - The simulation graph.  Each node is a (location,state) pair.
  llvm::OwningPtr<ExplodedGraph> G;

  /// WList - A set of queued nodes that need to be processed by the
  ///  worklist algorithm.  It is up to the implementation of WList to decide
  ///  the order that nodes are processed.
  WorkList* WList;

  /// BCounterFactory - A factory object for created BlockCounter objects.
  ///   These are used to record for key nodes in the ExplodedGraph the
  ///   number of times different CFGBlocks have been visited along a path.
  BlockCounter::Factory BCounterFactory;

  /// The locations where we stopped doing work because we visited a location
  ///  too many times.
  BlocksAborted blocksAborted;

  void generateNode(const ProgramPoint& Loc, const GRState* State,
                    ExplodedNode* Pred);

  void HandleBlockEdge(const BlockEdge& E, ExplodedNode* Pred);
  void HandleBlockEntrance(const BlockEntrance& E, ExplodedNode* Pred);
  void HandleBlockExit(const CFGBlock* B, ExplodedNode* Pred);
  void HandlePostStmt(const CFGBlock* B, unsigned StmtIdx, ExplodedNode *Pred);

  void HandleBranch(const Stmt* Cond, const Stmt* Term, const CFGBlock* B,
                    ExplodedNode* Pred);
  void HandleCallEnter(const CallEnter &L, const CFGBlock *Block,
                       unsigned Index, ExplodedNode *Pred);
  void HandleCallExit(const CallExit &L, ExplodedNode *Pred);

  /// Get the initial state from the subengine.
  const GRState* getInitialState(const LocationContext *InitLoc) {
    return SubEng.getInitialState(InitLoc);
  }

  void ProcessEndPath(EndPathNodeBuilder& Builder) {
    SubEng.ProcessEndPath(Builder);
  }

  void ProcessElement(const CFGElement E, StmtNodeBuilder& Builder) {
    SubEng.ProcessElement(E, Builder);
  }

  bool ProcessBlockEntrance(const CFGBlock* Blk, const ExplodedNode *Pred,
                            BlockCounter BC) {
    return SubEng.ProcessBlockEntrance(Blk, Pred, BC);
  }


  void ProcessBranch(const Stmt* Condition, const Stmt* Terminator,
                     BranchNodeBuilder& Builder) {
    SubEng.ProcessBranch(Condition, Terminator, Builder);
  }


  void ProcessIndirectGoto(IndirectGotoNodeBuilder& Builder) {
    SubEng.ProcessIndirectGoto(Builder);
  }


  void ProcessSwitch(SwitchNodeBuilder& Builder) {
    SubEng.ProcessSwitch(Builder);
  }

  void ProcessCallEnter(CallEnterNodeBuilder &Builder) {
    SubEng.ProcessCallEnter(Builder);
  }

  void ProcessCallExit(CallExitNodeBuilder &Builder) {
    SubEng.ProcessCallExit(Builder);
  }

private:
  CoreEngine(const CoreEngine&); // Do not implement.
  CoreEngine& operator=(const CoreEngine&);

public:
  /// Construct a CoreEngine object to analyze the provided CFG using
  ///  a DFS exploration of the exploded graph.
  CoreEngine(SubEngine& subengine)
    : SubEng(subengine), G(new ExplodedGraph()),
      WList(WorkList::MakeBFS()),
      BCounterFactory(G->getAllocator()) {}

  /// Construct a CoreEngine object to analyze the provided CFG and to
  ///  use the provided worklist object to execute the worklist algorithm.
  ///  The CoreEngine object assumes ownership of 'wlist'.
  CoreEngine(WorkList* wlist, SubEngine& subengine)
    : SubEng(subengine), G(new ExplodedGraph()), WList(wlist),
      BCounterFactory(G->getAllocator()) {}

  ~CoreEngine() {
    delete WList;
  }

  /// getGraph - Returns the exploded graph.
  ExplodedGraph& getGraph() { return *G.get(); }

  /// takeGraph - Returns the exploded graph.  Ownership of the graph is
  ///  transfered to the caller.
  ExplodedGraph* takeGraph() { return G.take(); }

  /// ExecuteWorkList - Run the worklist algorithm for a maximum number of
  ///  steps.  Returns true if there is still simulation state on the worklist.
  bool ExecuteWorkList(const LocationContext *L, unsigned Steps,
                       const GRState *InitState);
  void ExecuteWorkListWithInitialState(const LocationContext *L, unsigned Steps,
                                       const GRState *InitState, 
                                       ExplodedNodeSet &Dst);

  // Functions for external checking of whether we have unfinished work
  bool wasBlockAborted() const { return !blocksAborted.empty(); }
  bool hasWorkRemaining() const { return wasBlockAborted() || WList->hasWork(); }

  WorkList *getWorkList() const { return WList; }

  BlocksAborted::const_iterator blocks_aborted_begin() const {
    return blocksAborted.begin();
  }
  BlocksAborted::const_iterator blocks_aborted_end() const {
    return blocksAborted.end();
  }
};

class StmtNodeBuilder {
  CoreEngine& Eng;
  const CFGBlock& B;
  const unsigned Idx;
  ExplodedNode* Pred;
  GRStateManager& Mgr;

public:
  bool PurgingDeadSymbols;
  bool BuildSinks;
  bool HasGeneratedNode;
  ProgramPoint::Kind PointKind;
  const void *Tag;

  const GRState* CleanedState;


  typedef llvm::SmallPtrSet<ExplodedNode*,5> DeferredTy;
  DeferredTy Deferred;

  void GenerateAutoTransition(ExplodedNode* N);

public:
  StmtNodeBuilder(const CFGBlock* b, unsigned idx, ExplodedNode* N,
                    CoreEngine* e, GRStateManager &mgr);

  ~StmtNodeBuilder();

  ExplodedNode* getBasePredecessor() const { return Pred; }

  // FIXME: This should not be exposed.
  WorkList *getWorkList() { return Eng.WList; }

  void SetCleanedState(const GRState* St) {
    CleanedState = St;
  }

  BlockCounter getBlockCounter() const { return Eng.WList->getBlockCounter();}

  unsigned getCurrentBlockCount() const {
    return getBlockCounter().getNumVisited(
                            Pred->getLocationContext()->getCurrentStackFrame(),
                                           B.getBlockID());
  }

  ExplodedNode* generateNode(PostStmt PP,const GRState* St,ExplodedNode* Pred) {
    HasGeneratedNode = true;
    return generateNodeInternal(PP, St, Pred);
  }

  ExplodedNode* generateNode(const Stmt *S, const GRState *St,
                             ExplodedNode *Pred, ProgramPoint::Kind K) {
    HasGeneratedNode = true;

    if (PurgingDeadSymbols)
      K = ProgramPoint::PostPurgeDeadSymbolsKind;

    return generateNodeInternal(S, St, Pred, K, Tag);
  }

  ExplodedNode* generateNode(const Stmt *S, const GRState *St,
                             ExplodedNode *Pred) {
    return generateNode(S, St, Pred, PointKind);
  }

  ExplodedNode *generateNode(const ProgramPoint &PP, const GRState* State,
                             ExplodedNode* Pred) {
    HasGeneratedNode = true;
    return generateNodeInternal(PP, State, Pred);
  }

  ExplodedNode*
  generateNodeInternal(const ProgramPoint &PP, const GRState* State,
                       ExplodedNode* Pred);

  ExplodedNode*
  generateNodeInternal(const Stmt* S, const GRState* State, ExplodedNode* Pred,
                   ProgramPoint::Kind K = ProgramPoint::PostStmtKind,
                   const void *tag = 0);

  /// getStmt - Return the current block-level expression associated with
  ///  this builder.
  const Stmt* getStmt() const { 
    CFGStmt CS = B[Idx].getAs<CFGStmt>();
    if (CS)
      return CS.getStmt();
    else
      return 0;
  }

  /// getBlock - Return the CFGBlock associated with the block-level expression
  ///  of this builder.
  const CFGBlock* getBlock() const { return &B; }

  unsigned getIndex() const { return Idx; }

  const GRState* GetState(ExplodedNode* Pred) const {
    if (Pred == getBasePredecessor())
      return CleanedState;
    else
      return Pred->getState();
  }

  ExplodedNode* MakeNode(ExplodedNodeSet& Dst, const Stmt* S, 
                         ExplodedNode* Pred, const GRState* St) {
    return MakeNode(Dst, S, Pred, St, PointKind);
  }

  ExplodedNode* MakeNode(ExplodedNodeSet& Dst, const Stmt* S,ExplodedNode* Pred,
                         const GRState* St, ProgramPoint::Kind K);

  ExplodedNode* MakeSinkNode(ExplodedNodeSet& Dst, const Stmt* S,
                       ExplodedNode* Pred, const GRState* St) {
    bool Tmp = BuildSinks;
    BuildSinks = true;
    ExplodedNode* N = MakeNode(Dst, S, Pred, St);
    BuildSinks = Tmp;
    return N;
  }
};

class BranchNodeBuilder {
  CoreEngine& Eng;
  const CFGBlock* Src;
  const CFGBlock* DstT;
  const CFGBlock* DstF;
  ExplodedNode* Pred;

  typedef llvm::SmallVector<ExplodedNode*,3> DeferredTy;
  DeferredTy Deferred;

  bool GeneratedTrue;
  bool GeneratedFalse;
  bool InFeasibleTrue;
  bool InFeasibleFalse;

public:
  BranchNodeBuilder(const CFGBlock* src, const CFGBlock* dstT, 
                      const CFGBlock* dstF, ExplodedNode* pred, CoreEngine* e)
  : Eng(*e), Src(src), DstT(dstT), DstF(dstF), Pred(pred),
    GeneratedTrue(false), GeneratedFalse(false),
    InFeasibleTrue(!DstT), InFeasibleFalse(!DstF) {}

  ~BranchNodeBuilder();

  ExplodedNode* getPredecessor() const { return Pred; }

  const ExplodedGraph& getGraph() const { return *Eng.G; }

  BlockCounter getBlockCounter() const { return Eng.WList->getBlockCounter();}

  ExplodedNode* generateNode(const GRState* State, bool branch);

  const CFGBlock* getTargetBlock(bool branch) const {
    return branch ? DstT : DstF;
  }

  void markInfeasible(bool branch) {
    if (branch)
      InFeasibleTrue = GeneratedTrue = true;
    else
      InFeasibleFalse = GeneratedFalse = true;
  }

  bool isFeasible(bool branch) {
    return branch ? !InFeasibleTrue : !InFeasibleFalse;
  }

  const GRState* getState() const {
    return getPredecessor()->getState();
  }
};

class IndirectGotoNodeBuilder {
  CoreEngine& Eng;
  const CFGBlock* Src;
  const CFGBlock& DispatchBlock;
  const Expr* E;
  ExplodedNode* Pred;

public:
  IndirectGotoNodeBuilder(ExplodedNode* pred, const CFGBlock* src, 
                    const Expr* e, const CFGBlock* dispatch, CoreEngine* eng)
    : Eng(*eng), Src(src), DispatchBlock(*dispatch), E(e), Pred(pred) {}

  class iterator {
    CFGBlock::const_succ_iterator I;

    friend class IndirectGotoNodeBuilder;
    iterator(CFGBlock::const_succ_iterator i) : I(i) {}
  public:

    iterator& operator++() { ++I; return *this; }
    bool operator!=(const iterator& X) const { return I != X.I; }

    const LabelStmt* getLabel() const {
      return llvm::cast<LabelStmt>((*I)->getLabel());
    }

    const CFGBlock*  getBlock() const {
      return *I;
    }
  };

  iterator begin() { return iterator(DispatchBlock.succ_begin()); }
  iterator end() { return iterator(DispatchBlock.succ_end()); }

  ExplodedNode* generateNode(const iterator& I, const GRState* State,
                             bool isSink = false);

  const Expr* getTarget() const { return E; }

  const GRState* getState() const { return Pred->State; }
};

class SwitchNodeBuilder {
  CoreEngine& Eng;
  const CFGBlock* Src;
  const Expr* Condition;
  ExplodedNode* Pred;

public:
  SwitchNodeBuilder(ExplodedNode* pred, const CFGBlock* src,
                      const Expr* condition, CoreEngine* eng)
  : Eng(*eng), Src(src), Condition(condition), Pred(pred) {}

  class iterator {
    CFGBlock::const_succ_reverse_iterator I;

    friend class SwitchNodeBuilder;
    iterator(CFGBlock::const_succ_reverse_iterator i) : I(i) {}

  public:
    iterator& operator++() { ++I; return *this; }
    bool operator!=(const iterator &X) const { return I != X.I; }
    bool operator==(const iterator &X) const { return I == X.I; }

    const CaseStmt* getCase() const {
      return llvm::cast<CaseStmt>((*I)->getLabel());
    }

    const CFGBlock* getBlock() const {
      return *I;
    }
  };

  iterator begin() { return iterator(Src->succ_rbegin()+1); }
  iterator end() { return iterator(Src->succ_rend()); }

  const SwitchStmt *getSwitch() const {
    return llvm::cast<SwitchStmt>(Src->getTerminator());
  }

  ExplodedNode* generateCaseStmtNode(const iterator& I, const GRState* State);

  ExplodedNode* generateDefaultCaseNode(const GRState* State,
                                        bool isSink = false);

  const Expr* getCondition() const { return Condition; }

  const GRState* getState() const { return Pred->State; }
};

class EndPathNodeBuilder {
  CoreEngine &Eng;
  const CFGBlock& B;
  ExplodedNode* Pred;

public:
  bool HasGeneratedNode;

public:
  EndPathNodeBuilder(const CFGBlock* b, ExplodedNode* N, CoreEngine* e)
    : Eng(*e), B(*b), Pred(N), HasGeneratedNode(false) {}

  ~EndPathNodeBuilder();

  WorkList &getWorkList() { return *Eng.WList; }

  ExplodedNode* getPredecessor() const { return Pred; }

  BlockCounter getBlockCounter() const {
    return Eng.WList->getBlockCounter();
  }

  unsigned getCurrentBlockCount() const {
    return getBlockCounter().getNumVisited(
                            Pred->getLocationContext()->getCurrentStackFrame(),
                                           B.getBlockID());
  }

  ExplodedNode* generateNode(const GRState* State, const void *tag = 0,
                             ExplodedNode *P = 0);

  void GenerateCallExitNode(const GRState *state);

  const CFGBlock* getBlock() const { return &B; }

  const GRState* getState() const {
    return getPredecessor()->getState();
  }
};

class CallEnterNodeBuilder {
  CoreEngine &Eng;

  const ExplodedNode *Pred;

  // The call site. For implicit automatic object dtor, this is the trigger 
  // statement.
  const Stmt *CE;

  // The context of the callee.
  const StackFrameContext *CalleeCtx;

  // The parent block of the CallExpr.
  const CFGBlock *Block;

  // The CFGBlock index of the CallExpr.
  unsigned Index;

public:
  CallEnterNodeBuilder(CoreEngine &eng, const ExplodedNode *pred, 
                         const Stmt *s, const StackFrameContext *callee, 
                         const CFGBlock *blk, unsigned idx)
    : Eng(eng), Pred(pred), CE(s), CalleeCtx(callee), Block(blk), Index(idx) {}

  const GRState *getState() const { return Pred->getState(); }

  const LocationContext *getLocationContext() const { 
    return Pred->getLocationContext();
  }

  const Stmt *getCallExpr() const { return CE; }

  const StackFrameContext *getCalleeContext() const { return CalleeCtx; }

  const CFGBlock *getBlock() const { return Block; }

  unsigned getIndex() const { return Index; }

  void generateNode(const GRState *state);
};

class CallExitNodeBuilder {
  CoreEngine &Eng;
  const ExplodedNode *Pred;

public:
  CallExitNodeBuilder(CoreEngine &eng, const ExplodedNode *pred)
    : Eng(eng), Pred(pred) {}

  const ExplodedNode *getPredecessor() const { return Pred; }

  const GRState *getState() const { return Pred->getState(); }

  void generateNode(const GRState *state);
}; 

} // end GR namespace

} // end clang namespace

#endif
