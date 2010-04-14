//==- GRCoreEngine.h - Path-Sensitive Dataflow Engine --------------*- C++ -*-//
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

#ifndef LLVM_CLANG_ANALYSIS_GRENGINE
#define LLVM_CLANG_ANALYSIS_GRENGINE

#include "clang/AST/Expr.h"
#include "clang/Checker/PathSensitive/ExplodedGraph.h"
#include "clang/Checker/PathSensitive/GRWorkList.h"
#include "clang/Checker/PathSensitive/GRBlockCounter.h"
#include "clang/Checker/PathSensitive/GRAuditor.h"
#include "clang/Checker/PathSensitive/GRSubEngine.h"
#include "llvm/ADT/OwningPtr.h"

namespace clang {

//===----------------------------------------------------------------------===//
/// GRCoreEngine - Implements the core logic of the graph-reachability
///   analysis. It traverses the CFG and generates the ExplodedGraph.
///   Program "states" are treated as opaque void pointers.
///   The template class GRCoreEngine (which subclasses GRCoreEngine)
///   provides the matching component to the engine that knows the actual types
///   for states.  Note that this engine only dispatches to transfer functions
///   at the statement and block-level.  The analyses themselves must implement
///   any transfer function logic and the sub-expression level (if any).
class GRCoreEngine {
  friend class GRStmtNodeBuilder;
  friend class GRBranchNodeBuilder;
  friend class GRIndirectGotoNodeBuilder;
  friend class GRSwitchNodeBuilder;
  friend class GREndPathNodeBuilder;
  friend class GRCallEnterNodeBuilder;
  friend class GRCallExitNodeBuilder;

  GRSubEngine& SubEngine;

  /// G - The simulation graph.  Each node is a (location,state) pair.
  llvm::OwningPtr<ExplodedGraph> G;

  /// WList - A set of queued nodes that need to be processed by the
  ///  worklist algorithm.  It is up to the implementation of WList to decide
  ///  the order that nodes are processed.
  GRWorkList* WList;

  /// BCounterFactory - A factory object for created GRBlockCounter objects.
  ///   These are used to record for key nodes in the ExplodedGraph the
  ///   number of times different CFGBlocks have been visited along a path.
  GRBlockCounter::Factory BCounterFactory;

  void GenerateNode(const ProgramPoint& Loc, const GRState* State,
                    ExplodedNode* Pred);

  void HandleBlockEdge(const BlockEdge& E, ExplodedNode* Pred);
  void HandleBlockEntrance(const BlockEntrance& E, ExplodedNode* Pred);
  void HandleBlockExit(CFGBlock* B, ExplodedNode* Pred);
  void HandlePostStmt(const PostStmt& S, CFGBlock* B,
                      unsigned StmtIdx, ExplodedNode *Pred);

  void HandleBranch(Stmt* Cond, Stmt* Term, CFGBlock* B,
                    ExplodedNode* Pred);
  void HandleCallEnter(const CallEnter &L, const CFGBlock *Block,
                       unsigned Index, ExplodedNode *Pred);
  void HandleCallExit(const CallExit &L, ExplodedNode *Pred);

  /// Get the initial state from the subengine.
  const GRState* getInitialState(const LocationContext *InitLoc) {
    return SubEngine.getInitialState(InitLoc);
  }

  void ProcessEndPath(GREndPathNodeBuilder& Builder);

  void ProcessStmt(CFGElement E, GRStmtNodeBuilder& Builder);

  bool ProcessBlockEntrance(CFGBlock* Blk, const ExplodedNode *Pred,
                            GRBlockCounter BC);


  void ProcessBranch(Stmt* Condition, Stmt* Terminator,
                     GRBranchNodeBuilder& Builder);


  void ProcessIndirectGoto(GRIndirectGotoNodeBuilder& Builder);


  void ProcessSwitch(GRSwitchNodeBuilder& Builder);

  void ProcessCallEnter(GRCallEnterNodeBuilder &Builder);
  void ProcessCallExit(GRCallExitNodeBuilder &Builder);

private:
  GRCoreEngine(const GRCoreEngine&); // Do not implement.
  GRCoreEngine& operator=(const GRCoreEngine&);

public:
  /// Construct a GRCoreEngine object to analyze the provided CFG using
  ///  a DFS exploration of the exploded graph.
  GRCoreEngine(ASTContext& ctx, GRSubEngine& subengine)
    : SubEngine(subengine), G(new ExplodedGraph(ctx)),
      WList(GRWorkList::MakeBFS()),
      BCounterFactory(G->getAllocator()) {}

  /// Construct a GRCoreEngine object to analyze the provided CFG and to
  ///  use the provided worklist object to execute the worklist algorithm.
  ///  The GRCoreEngine object assumes ownership of 'wlist'.
  GRCoreEngine(ASTContext& ctx, GRWorkList* wlist, GRSubEngine& subengine)
    : SubEngine(subengine), G(new ExplodedGraph(ctx)), WList(wlist),
      BCounterFactory(G->getAllocator()) {}

  ~GRCoreEngine() {
    delete WList;
  }

  /// getGraph - Returns the exploded graph.
  ExplodedGraph& getGraph() { return *G.get(); }

  /// takeGraph - Returns the exploded graph.  Ownership of the graph is
  ///  transfered to the caller.
  ExplodedGraph* takeGraph() { return G.take(); }

  /// ExecuteWorkList - Run the worklist algorithm for a maximum number of
  ///  steps.  Returns true if there is still simulation state on the worklist.
  bool ExecuteWorkList(const LocationContext *L, unsigned Steps);
};

class GRStmtNodeBuilder {
  GRCoreEngine& Eng;
  CFGBlock& B;
  const unsigned Idx;
  ExplodedNode* Pred;
  GRStateManager& Mgr;
  GRAuditor* Auditor;

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
  GRStmtNodeBuilder(CFGBlock* b, unsigned idx, ExplodedNode* N,
                    GRCoreEngine* e, GRStateManager &mgr);

  ~GRStmtNodeBuilder();

  ExplodedNode* getBasePredecessor() const { return Pred; }

  // FIXME: This should not be exposed.
  GRWorkList *getWorkList() { return Eng.WList; }

  void SetCleanedState(const GRState* St) {
    CleanedState = St;
  }

  GRBlockCounter getBlockCounter() const { return Eng.WList->getBlockCounter();}

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
  Stmt* getStmt() const { return B[Idx]; }

  /// getBlock - Return the CFGBlock associated with the block-level expression
  ///  of this builder.
  CFGBlock* getBlock() const { return &B; }

  unsigned getIndex() const { return Idx; }

  void setAuditor(GRAuditor* A) { Auditor = A; }

  const GRState* GetState(ExplodedNode* Pred) const {
    if (Pred == getBasePredecessor())
      return CleanedState;
    else
      return Pred->getState();
  }

  ExplodedNode* MakeNode(ExplodedNodeSet& Dst, Stmt* S, ExplodedNode* Pred,
                   const GRState* St) {
    return MakeNode(Dst, S, Pred, St, PointKind);
  }

  ExplodedNode* MakeNode(ExplodedNodeSet& Dst, Stmt* S, ExplodedNode* Pred,
                         const GRState* St, ProgramPoint::Kind K);

  ExplodedNode* MakeSinkNode(ExplodedNodeSet& Dst, Stmt* S,
                       ExplodedNode* Pred, const GRState* St) {
    bool Tmp = BuildSinks;
    BuildSinks = true;
    ExplodedNode* N = MakeNode(Dst, S, Pred, St);
    BuildSinks = Tmp;
    return N;
  }
};

class GRBranchNodeBuilder {
  GRCoreEngine& Eng;
  CFGBlock* Src;
  CFGBlock* DstT;
  CFGBlock* DstF;
  ExplodedNode* Pred;

  typedef llvm::SmallVector<ExplodedNode*,3> DeferredTy;
  DeferredTy Deferred;

  bool GeneratedTrue;
  bool GeneratedFalse;
  bool InFeasibleTrue;
  bool InFeasibleFalse;

public:
  GRBranchNodeBuilder(CFGBlock* src, CFGBlock* dstT, CFGBlock* dstF,
                          ExplodedNode* pred, GRCoreEngine* e)
  : Eng(*e), Src(src), DstT(dstT), DstF(dstF), Pred(pred),
    GeneratedTrue(false), GeneratedFalse(false),
    InFeasibleTrue(!DstT), InFeasibleFalse(!DstF) {}

  ~GRBranchNodeBuilder();

  ExplodedNode* getPredecessor() const { return Pred; }

  const ExplodedGraph& getGraph() const { return *Eng.G; }

  GRBlockCounter getBlockCounter() const { return Eng.WList->getBlockCounter();}

  ExplodedNode* generateNode(const GRState* State, bool branch);

  CFGBlock* getTargetBlock(bool branch) const {
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

class GRIndirectGotoNodeBuilder {
  GRCoreEngine& Eng;
  CFGBlock* Src;
  CFGBlock& DispatchBlock;
  Expr* E;
  ExplodedNode* Pred;

public:
  GRIndirectGotoNodeBuilder(ExplodedNode* pred, CFGBlock* src, Expr* e,
                            CFGBlock* dispatch, GRCoreEngine* eng)
  : Eng(*eng), Src(src), DispatchBlock(*dispatch), E(e), Pred(pred) {}

  class iterator {
    CFGBlock::succ_iterator I;

    friend class GRIndirectGotoNodeBuilder;
    iterator(CFGBlock::succ_iterator i) : I(i) {}
  public:

    iterator& operator++() { ++I; return *this; }
    bool operator!=(const iterator& X) const { return I != X.I; }

    LabelStmt* getLabel() const {
      return llvm::cast<LabelStmt>((*I)->getLabel());
    }

    CFGBlock*  getBlock() const {
      return *I;
    }
  };

  iterator begin() { return iterator(DispatchBlock.succ_begin()); }
  iterator end() { return iterator(DispatchBlock.succ_end()); }

  ExplodedNode* generateNode(const iterator& I, const GRState* State,
                             bool isSink = false);

  Expr* getTarget() const { return E; }

  const GRState* getState() const { return Pred->State; }
};

class GRSwitchNodeBuilder {
  GRCoreEngine& Eng;
  CFGBlock* Src;
  Expr* Condition;
  ExplodedNode* Pred;

public:
  GRSwitchNodeBuilder(ExplodedNode* pred, CFGBlock* src,
                      Expr* condition, GRCoreEngine* eng)
  : Eng(*eng), Src(src), Condition(condition), Pred(pred) {}

  class iterator {
    CFGBlock::succ_reverse_iterator I;

    friend class GRSwitchNodeBuilder;
    iterator(CFGBlock::succ_reverse_iterator i) : I(i) {}

  public:
    iterator& operator++() { ++I; return *this; }
    bool operator!=(const iterator& X) const { return I != X.I; }

    CaseStmt* getCase() const {
      return llvm::cast<CaseStmt>((*I)->getLabel());
    }

    CFGBlock* getBlock() const {
      return *I;
    }
  };

  iterator begin() { return iterator(Src->succ_rbegin()+1); }
  iterator end() { return iterator(Src->succ_rend()); }

  ExplodedNode* generateCaseStmtNode(const iterator& I, const GRState* State);

  ExplodedNode* generateDefaultCaseNode(const GRState* State,
                                        bool isSink = false);

  Expr* getCondition() const { return Condition; }

  const GRState* getState() const { return Pred->State; }
};

class GREndPathNodeBuilder {
  GRCoreEngine &Eng;
  CFGBlock& B;
  ExplodedNode* Pred;

public:
  bool HasGeneratedNode;

public:
  GREndPathNodeBuilder(CFGBlock* b, ExplodedNode* N, GRCoreEngine* e)
    : Eng(*e), B(*b), Pred(N), HasGeneratedNode(false) {}

  ~GREndPathNodeBuilder();

  GRWorkList &getWorkList() { return *Eng.WList; }

  ExplodedNode* getPredecessor() const { return Pred; }

  GRBlockCounter getBlockCounter() const {
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

  CFGBlock* getBlock() const { return &B; }

  const GRState* getState() const {
    return getPredecessor()->getState();
  }
};

class GRCallEnterNodeBuilder {
  GRCoreEngine &Eng;

  const ExplodedNode *Pred;

  // The call site.
  const Stmt *CE;

  // The definition of callee.
  const FunctionDecl *FD;

  // The parent block of the CallExpr.
  const CFGBlock *Block;

  // The CFGBlock index of the CallExpr.
  unsigned Index;

public:
  GRCallEnterNodeBuilder(GRCoreEngine &eng, const ExplodedNode *pred, 
                         const Stmt *s, const FunctionDecl *fd, 
                         const CFGBlock *blk, unsigned idx)
    : Eng(eng), Pred(pred), CE(s), FD(fd), Block(blk), Index(idx) {}

  const GRState *getState() const { return Pred->getState(); }

  const LocationContext *getLocationContext() const { 
    return Pred->getLocationContext();
  }

  const Stmt *getCallExpr() const { return CE; }

  const FunctionDecl *getCallee() const { return FD; }

  const CFGBlock *getBlock() const { return Block; }

  unsigned getIndex() const { return Index; }

  void GenerateNode(const GRState *state, const LocationContext *LocCtx);
};

class GRCallExitNodeBuilder {
  GRCoreEngine &Eng;
  const ExplodedNode *Pred;

public:
  GRCallExitNodeBuilder(GRCoreEngine &eng, const ExplodedNode *pred)
    : Eng(eng), Pred(pred) {}

  const ExplodedNode *getPredecessor() const { return Pred; }

  const GRState *getState() const { return Pred->getState(); }

  void GenerateNode(const GRState *state);
}; 
} // end clang namespace

#endif
