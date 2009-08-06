//==- GRCoreEngine.h - Path-Sensitive Dataflow Engine ------------------*- C++ -*-//
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
#include "clang/Analysis/PathSensitive/ExplodedGraph.h"
#include "clang/Analysis/PathSensitive/GRWorkList.h"
#include "clang/Analysis/PathSensitive/GRBlockCounter.h"
#include "clang/Analysis/PathSensitive/GRAuditor.h"
#include "clang/Analysis/PathSensitive/GRSubEngine.h"
#include "llvm/ADT/OwningPtr.h"

namespace clang {

class GRState;
class GRStateManager;

class GRStmtNodeBuilderImpl;
template<typename STATE> class GRStmtNodeBuilder;
class GRBranchNodeBuilderImpl;
template<typename STATE> class GRBranchNodeBuilder;
class GRIndirectGotoNodeBuilderImpl;
template<typename STATE> class GRIndirectGotoNodeBuilder;
class GRSwitchNodeBuilderImpl;
template<typename STATE> class GRSwitchNodeBuilder;
class GREndPathNodeBuilderImpl;
template<typename STATE> class GREndPathNodeBuilder;

class GRWorkList;
class GRCoreEngine;
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
public:
  typedef GRState                       StateTy;
  typedef GRStateManager                StateManagerTy;
  typedef ExplodedGraph                 GraphTy;
  typedef GraphTy::NodeTy               NodeTy;

private:
  friend class GRStmtNodeBuilderImpl;
  friend class GRBranchNodeBuilderImpl;
  friend class GRIndirectGotoNodeBuilderImpl;
  friend class GRSwitchNodeBuilderImpl;
  friend class GREndPathNodeBuilderImpl;

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

  /// Get the initial state from the subengine.
  const GRState* getInitialState() { 
    return SubEngine.getInitialState();
  }

  void ProcessEndPath(GREndPathNodeBuilderImpl& BuilderImpl);
  
  void ProcessStmt(Stmt* S, GRStmtNodeBuilderImpl& BuilderImpl);

  
  bool ProcessBlockEntrance(CFGBlock* Blk, const GRState* State,
                            GRBlockCounter BC);

  
  void ProcessBranch(Stmt* Condition, Stmt* Terminator,
                     GRBranchNodeBuilderImpl& BuilderImpl);


  void ProcessIndirectGoto(GRIndirectGotoNodeBuilderImpl& BuilderImpl);

  
  void ProcessSwitch(GRSwitchNodeBuilderImpl& BuilderImpl);

private:
  GRCoreEngine(const GRCoreEngine&); // Do not implement.
  GRCoreEngine& operator=(const GRCoreEngine&);
  
public:
  /// Construct a GRCoreEngine object to analyze the provided CFG using
  ///  a DFS exploration of the exploded graph.
  GRCoreEngine(CFG& cfg, Decl& cd, ASTContext& ctx, GRSubEngine& subengine)
    : SubEngine(subengine), G(new GraphTy(cfg, cd, ctx)), 
      WList(GRWorkList::MakeBFS()),
      BCounterFactory(G->getAllocator()) {}

  /// Construct a GRCoreEngine object to analyze the provided CFG and to
  ///  use the provided worklist object to execute the worklist algorithm.
  ///  The GRCoreEngine object assumes ownership of 'wlist'.
  GRCoreEngine(CFG& cfg, Decl& cd, ASTContext& ctx, GRWorkList* wlist,
               GRSubEngine& subengine)
    : SubEngine(subengine), G(new GraphTy(cfg, cd, ctx)), WList(wlist),
      BCounterFactory(G->getAllocator()) {}

  ~GRCoreEngine() {
    delete WList;
  }

  /// getGraph - Returns the exploded graph.
  GraphTy& getGraph() { return *G.get(); }
  
  /// takeGraph - Returns the exploded graph.  Ownership of the graph is
  ///  transfered to the caller.
  GraphTy* takeGraph() { return G.take(); }

  /// ExecuteWorkList - Run the worklist algorithm for a maximum number of
  ///  steps.  Returns true if there is still simulation state on the worklist.
  bool ExecuteWorkList(unsigned Steps);
  
  CFG& getCFG() { return G->getCFG(); }
};
  
class GRStmtNodeBuilderImpl {
  GRCoreEngine& Eng;
  CFGBlock& B;
  const unsigned Idx;
  ExplodedNode* Pred;
  ExplodedNode* LastNode;  
  
  typedef llvm::SmallPtrSet<ExplodedNode*,5> DeferredTy;
  DeferredTy Deferred;
  
  void GenerateAutoTransition(ExplodedNode* N);
  
public:
  GRStmtNodeBuilderImpl(CFGBlock* b, unsigned idx,
                    ExplodedNode* N, GRCoreEngine* e);      
  
  ~GRStmtNodeBuilderImpl();
  
  ExplodedNode* getBasePredecessor() const { return Pred; }
  
  ExplodedNode* getLastNode() const {
    return LastNode ? (LastNode->isSink() ? NULL : LastNode) : NULL;
  }
  
  GRBlockCounter getBlockCounter() const { return Eng.WList->getBlockCounter();}
  
  unsigned getCurrentBlockCount() const {
    return getBlockCounter().getNumVisited(B.getBlockID());
  }  
    
  ExplodedNode*
  generateNodeImpl(const ProgramPoint &PP, const GRState* State,
                   ExplodedNode* Pred);
  
  ExplodedNode*
  generateNodeImpl(const Stmt* S, const GRState* State, ExplodedNode* Pred,
                   ProgramPoint::Kind K = ProgramPoint::PostStmtKind,
                   const void *tag = 0);

  ExplodedNode*
  generateNodeImpl(const Stmt* S, const GRState* State,
                   ProgramPoint::Kind K = ProgramPoint::PostStmtKind,
                   const void *tag = 0) {
    ExplodedNode* N = getLastNode();
    assert (N && "Predecessor of new node is infeasible.");
    return generateNodeImpl(S, State, N, K, tag);
  }
  
  ExplodedNode*
  generateNodeImpl(const Stmt* S, const GRState* State, const void *tag = 0) {
    ExplodedNode* N = getLastNode();
    assert (N && "Predecessor of new node is infeasible.");
    return generateNodeImpl(S, State, N, ProgramPoint::PostStmtKind, tag);
  }
  
  /// getStmt - Return the current block-level expression associated with
  ///  this builder.
  Stmt* getStmt() const { return B[Idx]; }
  
  /// getBlock - Return the CFGBlock associated with the block-level expression
  ///  of this builder.
  CFGBlock* getBlock() const { return &B; }
};
  
  
template<typename STATE>
class GRStmtNodeBuilder  {
public:
  typedef STATE                       StateTy;
  typedef typename StateTy::ManagerTy StateManagerTy;
  typedef ExplodedNode       NodeTy;
  
private:
  GRStmtNodeBuilderImpl& NB;
  StateManagerTy& Mgr;
  const StateTy* CleanedState;  
  GRAuditor<StateTy>* Auditor;
  
public:
  GRStmtNodeBuilder(GRStmtNodeBuilderImpl& nb, StateManagerTy& mgr) :
    NB(nb), Mgr(mgr), Auditor(0), PurgingDeadSymbols(false),
    BuildSinks(false), HasGeneratedNode(false),
    PointKind(ProgramPoint::PostStmtKind), Tag(0) {
      
    CleanedState = getLastNode()->getState();
  }

  void setAuditor(GRAuditor<StateTy>* A) {
    Auditor = A;
  }
    
  NodeTy* getLastNode() const {
    return static_cast<NodeTy*>(NB.getLastNode());
  }
  
  NodeTy* generateNode(PostStmt PP, const StateTy* St, NodeTy* Pred) {
    HasGeneratedNode = true;
    return static_cast<NodeTy*>(NB.generateNodeImpl(PP, St, Pred));
  }
  
  NodeTy* generateNode(const Stmt* S, const StateTy* St, NodeTy* Pred,
                       ProgramPoint::Kind K) {
    HasGeneratedNode = true;
    if (PurgingDeadSymbols) K = ProgramPoint::PostPurgeDeadSymbolsKind;      
    return static_cast<NodeTy*>(NB.generateNodeImpl(S, St, Pred, K, Tag));
  }
  
  NodeTy* generateNode(const Stmt* S, const StateTy* St, NodeTy* Pred) {
    return generateNode(S, St, Pred, PointKind);
  }
  
  NodeTy* generateNode(const Stmt* S, const StateTy* St, ProgramPoint::Kind K) {
    HasGeneratedNode = true;
    if (PurgingDeadSymbols) K = ProgramPoint::PostPurgeDeadSymbolsKind;      
    return static_cast<NodeTy*>(NB.generateNodeImpl(S, St, K, Tag));
  }
  
  NodeTy* generateNode(const Stmt* S, const StateTy* St) {
    return generateNode(S, St, PointKind);
  }

  
  GRBlockCounter getBlockCounter() const {
    return NB.getBlockCounter();
  }  
  
  unsigned getCurrentBlockCount() const {
    return NB.getCurrentBlockCount();
  }
  
  const StateTy* GetState(NodeTy* Pred) const {
    if ((ExplodedNode*) Pred == NB.getBasePredecessor())
      return CleanedState;
    else
      return Pred->getState();
  }
  
  void SetCleanedState(const StateTy* St) {
    CleanedState = St;
  }
  
  NodeTy* MakeNode(ExplodedNodeSet& Dst, Stmt* S,
                   NodeTy* Pred, const StateTy* St) {
    return MakeNode(Dst, S, Pred, St, PointKind);
  }
  
  NodeTy* MakeNode(ExplodedNodeSet& Dst, Stmt* S,
                   NodeTy* Pred, const StateTy* St, ProgramPoint::Kind K) {    
    
    const StateTy* PredState = GetState(Pred);
        
    // If the state hasn't changed, don't generate a new node.
    if (!BuildSinks && St == PredState && Auditor == 0) {
      Dst.Add(Pred);
      return NULL;
    }
    
    NodeTy* N = generateNode(S, St, Pred, K);
    
    if (N) {      
      if (BuildSinks)
        N->markAsSink();
      else {
        if (Auditor && Auditor->Audit(N, Mgr))
          N->markAsSink();
        
        Dst.Add(N);
      }
    }
    
    return N;
  }
  
  NodeTy* MakeSinkNode(ExplodedNodeSet& Dst, Stmt* S,
                       NodeTy* Pred, const StateTy* St) { 
    bool Tmp = BuildSinks;
    BuildSinks = true;
    NodeTy* N = MakeNode(Dst, S, Pred, St);
    BuildSinks = Tmp;
    return N;
  }
  
  bool PurgingDeadSymbols;
  bool BuildSinks;
  bool HasGeneratedNode;
  ProgramPoint::Kind PointKind;
  const void *Tag;
};
  
class GRBranchNodeBuilderImpl {
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
  GRBranchNodeBuilderImpl(CFGBlock* src, CFGBlock* dstT, CFGBlock* dstF,
                          ExplodedNode* pred, GRCoreEngine* e) 
  : Eng(*e), Src(src), DstT(dstT), DstF(dstF), Pred(pred),
    GeneratedTrue(false), GeneratedFalse(false),
    InFeasibleTrue(!DstT), InFeasibleFalse(!DstF) {}
  
  ~GRBranchNodeBuilderImpl();
  
  ExplodedNode* getPredecessor() const { return Pred; }
  const ExplodedGraph& getGraph() const { return *Eng.G; }
  GRBlockCounter getBlockCounter() const { return Eng.WList->getBlockCounter();}
    
  ExplodedNode* generateNodeImpl(const GRState* State, bool branch);
  
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
};

template<typename STATE>
class GRBranchNodeBuilder {
  typedef STATE                                  StateTy;
  typedef ExplodedGraph                          GraphTy;
  typedef typename GraphTy::NodeTy               NodeTy;
  
  GRBranchNodeBuilderImpl& NB;
  
public:
  GRBranchNodeBuilder(GRBranchNodeBuilderImpl& nb) : NB(nb) {}
  
  const GraphTy& getGraph() const {
    return static_cast<const GraphTy&>(NB.getGraph());
  }
  
  NodeTy* getPredecessor() const {
    return static_cast<NodeTy*>(NB.getPredecessor());
  }
  
  const StateTy* getState() const {
    return getPredecessor()->getState();
  }

  NodeTy* generateNode(const StateTy* St, bool branch) {
    return static_cast<NodeTy*>(NB.generateNodeImpl(St, branch));
  }
  
  GRBlockCounter getBlockCounter() const {
    return NB.getBlockCounter();
  }
  
  CFGBlock* getTargetBlock(bool branch) const {
    return NB.getTargetBlock(branch);
  }
  
  void markInfeasible(bool branch) {
    NB.markInfeasible(branch);
  }
  
  bool isFeasible(bool branch) {
    return NB.isFeasible(branch);
  }
};
  
class GRIndirectGotoNodeBuilderImpl {
  GRCoreEngine& Eng;
  CFGBlock* Src;
  CFGBlock& DispatchBlock;
  Expr* E;
  ExplodedNode* Pred;  
public:
  GRIndirectGotoNodeBuilderImpl(ExplodedNode* pred, CFGBlock* src,
                                Expr* e, CFGBlock* dispatch,
                                GRCoreEngine* eng)
  : Eng(*eng), Src(src), DispatchBlock(*dispatch), E(e), Pred(pred) {}
  

  class Iterator {
    CFGBlock::succ_iterator I;
    
    friend class GRIndirectGotoNodeBuilderImpl;    
    Iterator(CFGBlock::succ_iterator i) : I(i) {}    
  public:
    
    Iterator& operator++() { ++I; return *this; }
    bool operator!=(const Iterator& X) const { return I != X.I; }
    
    LabelStmt* getLabel() const {
      return llvm::cast<LabelStmt>((*I)->getLabel());
    }
    
    CFGBlock*  getBlock() const {
      return *I;
    }
  };
  
  Iterator begin() { return Iterator(DispatchBlock.succ_begin()); }
  Iterator end() { return Iterator(DispatchBlock.succ_end()); }
  
  ExplodedNode* generateNodeImpl(const Iterator& I, const GRState* State,
                                     bool isSink);
  
  Expr* getTarget() const { return E; }
  const void* getState() const { return Pred->State; }
};
  
template<typename STATE>
class GRIndirectGotoNodeBuilder {
  typedef STATE                                  StateTy;
  typedef ExplodedGraph                          GraphTy;
  typedef typename GraphTy::NodeTy               NodeTy;

  GRIndirectGotoNodeBuilderImpl& NB;

public:
  GRIndirectGotoNodeBuilder(GRIndirectGotoNodeBuilderImpl& nb) : NB(nb) {}
  
  typedef GRIndirectGotoNodeBuilderImpl::Iterator     iterator;

  iterator begin() { return NB.begin(); }
  iterator end() { return NB.end(); }
  
  Expr* getTarget() const { return NB.getTarget(); }
  
  NodeTy* generateNode(const iterator& I, const StateTy* St, bool isSink=false){    
    return static_cast<NodeTy*>(NB.generateNodeImpl(I, St, isSink));
  }
  
  const StateTy* getState() const {
    return static_cast<const StateTy*>(NB.getState());
  }    
};
  
class GRSwitchNodeBuilderImpl {
  GRCoreEngine& Eng;
  CFGBlock* Src;
  Expr* Condition;
  ExplodedNode* Pred;  
public:
  GRSwitchNodeBuilderImpl(ExplodedNode* pred, CFGBlock* src,
                          Expr* condition, GRCoreEngine* eng)
  : Eng(*eng), Src(src), Condition(condition), Pred(pred) {}
  
  class Iterator {
    CFGBlock::succ_reverse_iterator I;
    
    friend class GRSwitchNodeBuilderImpl;    
    Iterator(CFGBlock::succ_reverse_iterator i) : I(i) {}    
  public:
    
    Iterator& operator++() { ++I; return *this; }
    bool operator!=(const Iterator& X) const { return I != X.I; }
    
    CaseStmt* getCase() const {
      return llvm::cast<CaseStmt>((*I)->getLabel());
    }
    
    CFGBlock* getBlock() const {
      return *I;
    }
  };
  
  Iterator begin() { return Iterator(Src->succ_rbegin()+1); }
  Iterator end() { return Iterator(Src->succ_rend()); }
  
  ExplodedNode* generateCaseStmtNodeImpl(const Iterator& I,
                                             const GRState* State);
  
  ExplodedNode* generateDefaultCaseNodeImpl(const GRState* State,
                                                bool isSink);
  
  Expr* getCondition() const { return Condition; }
  const void* getState() const { return Pred->State; }
};

template<typename STATE>
class GRSwitchNodeBuilder {
  typedef STATE                                  StateTy;
  typedef ExplodedGraph                          GraphTy;
  typedef typename GraphTy::NodeTy               NodeTy;
  
  GRSwitchNodeBuilderImpl& NB;
  
public:
  GRSwitchNodeBuilder(GRSwitchNodeBuilderImpl& nb) : NB(nb) {}
  
  typedef GRSwitchNodeBuilderImpl::Iterator     iterator;
  
  iterator begin() { return NB.begin(); }
  iterator end() { return NB.end(); }
  
  Expr* getCondition() const { return NB.getCondition(); }
  
  NodeTy* generateCaseStmtNode(const iterator& I, const StateTy* St) {
    return static_cast<NodeTy*>(NB.generateCaseStmtNodeImpl(I, St));
  }
  
  NodeTy* generateDefaultCaseNode(const StateTy* St, bool isSink = false) {    
    return static_cast<NodeTy*>(NB.generateDefaultCaseNodeImpl(St, isSink));
  }
  
  const StateTy* getState() const {
    return static_cast<const StateTy*>(NB.getState());
  }    
};
  

class GREndPathNodeBuilderImpl {
  GRCoreEngine& Eng;
  CFGBlock& B;
  ExplodedNode* Pred;  
  bool HasGeneratedNode;
  
public:
  GREndPathNodeBuilderImpl(CFGBlock* b, ExplodedNode* N,
                           GRCoreEngine* e)
    : Eng(*e), B(*b), Pred(N), HasGeneratedNode(false) {}      
  
  ~GREndPathNodeBuilderImpl();
  
  ExplodedNode* getPredecessor() const { return Pred; }
    
  GRBlockCounter getBlockCounter() const { return Eng.WList->getBlockCounter();}
  
  unsigned getCurrentBlockCount() const {
    return getBlockCounter().getNumVisited(B.getBlockID());
  }  
  
  ExplodedNode* generateNodeImpl(const GRState* State,
                                     const void *tag = 0,
                                     ExplodedNode *P = 0);
    
  CFGBlock* getBlock() const { return &B; }
};


template<typename STATE>
class GREndPathNodeBuilder  {
  typedef STATE                   StateTy;
  typedef ExplodedNode   NodeTy;
  
  GREndPathNodeBuilderImpl& NB;
  
public:
  GREndPathNodeBuilder(GREndPathNodeBuilderImpl& nb) : NB(nb) {}
  
  NodeTy* getPredecessor() const {
    return static_cast<NodeTy*>(NB.getPredecessor());
  }
  
  GRBlockCounter getBlockCounter() const {
    return NB.getBlockCounter();
  }  
  
  unsigned getCurrentBlockCount() const {
    return NB.getCurrentBlockCount();
  }
  
  const StateTy* getState() const {
    return getPredecessor()->getState();
  }
  
  NodeTy* MakeNode(const StateTy* St, const void *tag = 0) {  
    return static_cast<NodeTy*>(NB.generateNodeImpl(St, tag));
  }
  
  NodeTy *generateNode(const StateTy *St, NodeTy *Pred, const void *tag = 0) {
    return static_cast<NodeTy*>(NB.generateNodeImpl(St, tag, Pred));
  }                                
};

} // end clang namespace

#endif
