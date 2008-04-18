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
#include "llvm/ADT/OwningPtr.h"

namespace clang {
  
class GRStmtNodeBuilderImpl;
class GRBranchNodeBuilderImpl;
class GRIndirectGotoNodeBuilderImpl;
class GRSwitchNodeBuilderImpl;
class GREndPathNodeBuilderImpl;
class GRWorkList;

//===----------------------------------------------------------------------===//
/// GRCoreEngineImpl - Implements the core logic of the graph-reachability analysis.
///   It traverses the CFG and generates the ExplodedGraph. Program "states"
///   are treated as opaque void pointers.  The template class GRCoreEngine
///   (which subclasses GRCoreEngineImpl) provides the matching component
///   to the engine that knows the actual types for states.  Note that this
///   engine only dispatches to transfer functions as the statement and
///   block-level.  The analyses themselves must implement any transfer
///   function logic and the sub-expression level (if any).
class GRCoreEngineImpl {
protected:
  friend class GRStmtNodeBuilderImpl;
  friend class GRBranchNodeBuilderImpl;
  friend class GRIndirectGotoNodeBuilderImpl;
  friend class GRSwitchNodeBuilderImpl;
  friend class GREndPathNodeBuilderImpl;
  
  typedef llvm::DenseMap<Stmt*,Stmt*> ParentMapTy;
    
  /// G - The simulation graph.  Each node is a (location,state) pair.
  llvm::OwningPtr<ExplodedGraphImpl> G;
  
  /// ParentMap - A lazily populated map from a Stmt* to its parent Stmt*.
  void* ParentMap;
  
  /// CurrentBlkExpr - The current Block-level expression being processed.
  ///  This is used when lazily populating ParentMap.
  Stmt* CurrentBlkExpr;
  
  /// WList - A set of queued nodes that need to be processed by the
  ///  worklist algorithm.  It is up to the implementation of WList to decide
  ///  the order that nodes are processed.
  GRWorkList* WList;
  
  /// BCounterFactory - A factory object for created GRBlockCounter objects.
  ///   These are used to record for key nodes in the ExplodedGraph the
  ///   number of times different CFGBlocks have been visited along a path.
  GRBlockCounter::Factory BCounterFactory;
  
  void GenerateNode(const ProgramPoint& Loc, void* State,
                    ExplodedNodeImpl* Pred = NULL);
  
  /// getInitialState - Gets the void* representing the initial 'state'
  ///  of the analysis.  This is simply a wrapper (implemented
  ///  in GRCoreEngine) that performs type erasure on the initial
  ///  state returned by the checker object.
  virtual void* getInitialState() = 0;
  
  void HandleBlockEdge(const BlockEdge& E, ExplodedNodeImpl* Pred);
  void HandleBlockEntrance(const BlockEntrance& E, ExplodedNodeImpl* Pred);
  void HandleBlockExit(CFGBlock* B, ExplodedNodeImpl* Pred);
  void HandlePostStmt(const PostStmt& S, CFGBlock* B,
                      unsigned StmtIdx, ExplodedNodeImpl *Pred);
  
  void HandleBranch(Expr* Cond, Stmt* Term, CFGBlock* B,
                    ExplodedNodeImpl* Pred);  
  
  virtual void ProcessEndPath(GREndPathNodeBuilderImpl& Builder) = 0;  
  
  virtual bool ProcessBlockEntrance(CFGBlock* Blk, void* State,
                                    GRBlockCounter BC) = 0;

  virtual void ProcessStmt(Stmt* S, GRStmtNodeBuilderImpl& Builder) = 0;

  virtual void ProcessBranch(Expr* Condition, Stmt* Terminator,
                             GRBranchNodeBuilderImpl& Builder) = 0;

  virtual void ProcessIndirectGoto(GRIndirectGotoNodeBuilderImpl& Builder) = 0;
  
  virtual void ProcessSwitch(GRSwitchNodeBuilderImpl& Builder) = 0;

private:
  GRCoreEngineImpl(const GRCoreEngineImpl&); // Do not implement.
  GRCoreEngineImpl& operator=(const GRCoreEngineImpl&);
  
protected:  
  GRCoreEngineImpl(ExplodedGraphImpl* g, GRWorkList* wl)
    : G(g), WList(wl), BCounterFactory(g->getAllocator()) {}
  
public:
  /// ExecuteWorkList - Run the worklist algorithm for a maximum number of
  ///  steps.  Returns true if there is still simulation state on the worklist.
  bool ExecuteWorkList(unsigned Steps);
  
  virtual ~GRCoreEngineImpl() {}
  
  CFG& getCFG() { return G->getCFG(); }
};
  
class GRStmtNodeBuilderImpl {
  GRCoreEngineImpl& Eng;
  CFGBlock& B;
  const unsigned Idx;
  ExplodedNodeImpl* Pred;
  ExplodedNodeImpl* LastNode;  
  bool HasGeneratedNode;
  bool Populated;
  
  typedef llvm::SmallPtrSet<ExplodedNodeImpl*,5> DeferredTy;
  DeferredTy Deferred;
  
  void GenerateAutoTransition(ExplodedNodeImpl* N);
  
public:
  GRStmtNodeBuilderImpl(CFGBlock* b, unsigned idx,
                    ExplodedNodeImpl* N, GRCoreEngineImpl* e);      
  
  ~GRStmtNodeBuilderImpl();
  
  ExplodedNodeImpl* getBasePredecessor() const { return Pred; }
  
  ExplodedNodeImpl* getLastNode() const {
    return LastNode ? (LastNode->isSink() ? NULL : LastNode) : NULL;
  }
  
  GRBlockCounter getBlockCounter() const { return Eng.WList->getBlockCounter();}
  
  unsigned getCurrentBlockCount() const {
    return getBlockCounter().getNumVisited(B.getBlockID());
  }  
  
  ExplodedNodeImpl* generateNodeImpl(Stmt* S, void* State,
                                     ExplodedNodeImpl* Pred);

  inline ExplodedNodeImpl* generateNodeImpl(Stmt* S, void* State) {
    ExplodedNodeImpl* N = getLastNode();
    assert (N && "Predecessor of new node is infeasible.");
    return generateNodeImpl(S, State, N);
  }
  
  Stmt* getStmt() const { return B[Idx]; }
  
  CFGBlock* getBlock() const { return &B; }
  
  bool hasGeneratedNode() const { return HasGeneratedNode; }
};
  
  
template<typename STATE>
class GRStmtNodeBuilder  {
  typedef STATE                   StateTy;
  typedef ExplodedNode<StateTy>   NodeTy;
  
  GRStmtNodeBuilderImpl& NB;
  StateTy* CleanedState;
  
  GRAuditor<StateTy> **CallExprAuditBeg, **CallExprAuditEnd;
  GRAuditor<StateTy> **ObjCMsgExprAuditBeg, **ObjCMsgExprAuditEnd;
  
public:
  GRStmtNodeBuilder(GRStmtNodeBuilderImpl& nb) : NB(nb),
    CallExprAuditBeg(0), CallExprAuditEnd(0),
    ObjCMsgExprAuditBeg(0), ObjCMsgExprAuditEnd(0),  BuildSinks(false) {
      
    CleanedState = getLastNode()->getState();
  }
  
  bool hasGeneratedNode() const { return NB.hasGeneratedNode(); }
  
  void setObjCMsgExprAuditors(GRAuditor<StateTy> **B,
                              GRAuditor<StateTy> **E) {
    ObjCMsgExprAuditBeg = B;
    ObjCMsgExprAuditEnd = E;
  }
  
  void setCallExprAuditors(GRAuditor<StateTy> **B,
                           GRAuditor<StateTy> **E) {
    CallExprAuditBeg = B;
    CallExprAuditEnd = E;
  }  
    
  NodeTy* getLastNode() const {
    return static_cast<NodeTy*>(NB.getLastNode());
  }
  
  NodeTy* generateNode(Stmt* S, StateTy* St, NodeTy* Pred) {
    return static_cast<NodeTy*>(NB.generateNodeImpl(S, St, Pred));
  }
  
  NodeTy* generateNode(Stmt* S, StateTy* St) {
    return static_cast<NodeTy*>(NB.generateNodeImpl(S, St));    
  }
  
  GRBlockCounter getBlockCounter() const {
    return NB.getBlockCounter();
  }  
  
  unsigned getCurrentBlockCount() const {
    return NB.getCurrentBlockCount();
  }
  
  StateTy* GetState(NodeTy* Pred) const {
    if ((ExplodedNodeImpl*) Pred == NB.getBasePredecessor())
      return CleanedState;
    else
      return Pred->getState();
  }
  
  void SetCleanedState(StateTy* St) {
    CleanedState = St;
  }
  
  NodeTy* MakeNode(ExplodedNodeSet<StateTy>& Dst, Stmt* S,
                 NodeTy* Pred, StateTy* St) {    
    
    StateTy* PredState = GetState(Pred);
    
    GRAuditor<StateTy> **AB = NULL, **AE = NULL;
    
    switch (S->getStmtClass()) {
      default: break;
      case Stmt::CallExprClass:
        AB = CallExprAuditBeg;
        AE = CallExprAuditEnd;
        break;
      case Stmt::ObjCMessageExprClass:
        AB = ObjCMsgExprAuditBeg;
        AE = ObjCMsgExprAuditEnd;
        break;
    }
    
    // If the state hasn't changed, don't generate a new node.
    if (!BuildSinks && St == PredState && AB == NULL) {
      Dst.Add(Pred);
      return NULL;
    }
    
    NodeTy* N = generateNode(S, St, Pred);
    
    if (N) {      
      if (BuildSinks)
        N->markAsSink();
      else {
        Dst.Add(N);
        
        for ( ; AB != AE; ++AB)
          if ((*AB)->Audit(N))
            N->markAsSink();            
      }
    }
    
    return N;
  }
  
  bool BuildSinks;  
};
  
class GRBranchNodeBuilderImpl {
  GRCoreEngineImpl& Eng;
  CFGBlock* Src;
  CFGBlock* DstT;
  CFGBlock* DstF;
  ExplodedNodeImpl* Pred;

  typedef llvm::SmallVector<ExplodedNodeImpl*,3> DeferredTy;
  DeferredTy Deferred;
  
  bool GeneratedTrue;
  bool GeneratedFalse;
  
public:
  GRBranchNodeBuilderImpl(CFGBlock* src, CFGBlock* dstT, CFGBlock* dstF,
                          ExplodedNodeImpl* pred, GRCoreEngineImpl* e) 
  : Eng(*e), Src(src), DstT(dstT), DstF(dstF), Pred(pred),
    GeneratedTrue(false), GeneratedFalse(false) {}
  
  ~GRBranchNodeBuilderImpl();
  
  ExplodedNodeImpl* getPredecessor() const { return Pred; }
  const ExplodedGraphImpl& getGraph() const { return *Eng.G; }
  GRBlockCounter getBlockCounter() const { return Eng.WList->getBlockCounter();}
    
  ExplodedNodeImpl* generateNodeImpl(void* State, bool branch);
  
  CFGBlock* getTargetBlock(bool branch) const {
    return branch ? DstT : DstF;
  }    
  
  void markInfeasible(bool branch) {
    if (branch) GeneratedTrue = true;
    else GeneratedFalse = true;
  }
};

template<typename STATE>
class GRBranchNodeBuilder {
  typedef STATE                                  StateTy;
  typedef ExplodedGraph<StateTy>                 GraphTy;
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
  
  StateTy* getState() const {
    return getPredecessor()->getState();
  }

  inline NodeTy* generateNode(StateTy* St, bool branch) {
    return static_cast<NodeTy*>(NB.generateNodeImpl(St, branch));
  }
  
  GRBlockCounter getBlockCounter() const {
    return NB.getBlockCounter();
  }
  
  CFGBlock* getTargetBlock(bool branch) const {
    return NB.getTargetBlock(branch);
  }
  
  inline void markInfeasible(bool branch) {
    NB.markInfeasible(branch);
  }
};
  
class GRIndirectGotoNodeBuilderImpl {
  GRCoreEngineImpl& Eng;
  CFGBlock* Src;
  CFGBlock& DispatchBlock;
  Expr* E;
  ExplodedNodeImpl* Pred;  
public:
  GRIndirectGotoNodeBuilderImpl(ExplodedNodeImpl* pred, CFGBlock* src,
                                Expr* e, CFGBlock* dispatch,
                                GRCoreEngineImpl* eng)
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
  
  ExplodedNodeImpl* generateNodeImpl(const Iterator& I, void* State,
                                     bool isSink);
  
  inline Expr* getTarget() const { return E; }
  inline void* getState() const { return Pred->State; }
};
  
template<typename STATE>
class GRIndirectGotoNodeBuilder {
  typedef STATE                                  StateTy;
  typedef ExplodedGraph<StateTy>                 GraphTy;
  typedef typename GraphTy::NodeTy               NodeTy;

  GRIndirectGotoNodeBuilderImpl& NB;

public:
  GRIndirectGotoNodeBuilder(GRIndirectGotoNodeBuilderImpl& nb) : NB(nb) {}
  
  typedef GRIndirectGotoNodeBuilderImpl::Iterator     iterator;

  inline iterator begin() { return NB.begin(); }
  inline iterator end() { return NB.end(); }
  
  inline Expr* getTarget() const { return NB.getTarget(); }
  
  inline NodeTy* generateNode(const iterator& I, StateTy* St, bool isSink=false){    
    return static_cast<NodeTy*>(NB.generateNodeImpl(I, St, isSink));
  }
  
  inline StateTy* getState() const {
    return static_cast<StateTy*>(NB.getState());
  }    
};
  
class GRSwitchNodeBuilderImpl {
  GRCoreEngineImpl& Eng;
  CFGBlock* Src;
  Expr* Condition;
  ExplodedNodeImpl* Pred;  
public:
  GRSwitchNodeBuilderImpl(ExplodedNodeImpl* pred, CFGBlock* src,
                          Expr* condition, GRCoreEngineImpl* eng)
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
  
  ExplodedNodeImpl* generateCaseStmtNodeImpl(const Iterator& I, void* State);
  ExplodedNodeImpl* generateDefaultCaseNodeImpl(void* State, bool isSink);
  
  inline Expr* getCondition() const { return Condition; }
  inline void* getState() const { return Pred->State; }
};

template<typename STATE>
class GRSwitchNodeBuilder {
  typedef STATE                                  StateTy;
  typedef ExplodedGraph<StateTy>                 GraphTy;
  typedef typename GraphTy::NodeTy               NodeTy;
  
  GRSwitchNodeBuilderImpl& NB;
  
public:
  GRSwitchNodeBuilder(GRSwitchNodeBuilderImpl& nb) : NB(nb) {}
  
  typedef GRSwitchNodeBuilderImpl::Iterator     iterator;
  
  inline iterator begin() { return NB.begin(); }
  inline iterator end() { return NB.end(); }
  
  inline Expr* getCondition() const { return NB.getCondition(); }
  
  inline NodeTy* generateCaseStmtNode(const iterator& I, StateTy* St) {
    return static_cast<NodeTy*>(NB.generateCaseStmtNodeImpl(I, St));
  }
  
  inline NodeTy* generateDefaultCaseNode(StateTy* St, bool isSink = false) {    
    return static_cast<NodeTy*>(NB.generateDefaultCaseNodeImpl(St, isSink));
  }
  
  inline StateTy* getState() const {
    return static_cast<StateTy*>(NB.getState());
  }    
};
  

class GREndPathNodeBuilderImpl {
  GRCoreEngineImpl& Eng;
  CFGBlock& B;
  ExplodedNodeImpl* Pred;  
  bool HasGeneratedNode;
  
public:
  GREndPathNodeBuilderImpl(CFGBlock* b, ExplodedNodeImpl* N,
                           GRCoreEngineImpl* e)
    : Eng(*e), B(*b), Pred(N), HasGeneratedNode(false) {}      
  
  ~GREndPathNodeBuilderImpl();
  
  ExplodedNodeImpl* getPredecessor() const { return Pred; }
    
  GRBlockCounter getBlockCounter() const { return Eng.WList->getBlockCounter();}
  
  unsigned getCurrentBlockCount() const {
    return getBlockCounter().getNumVisited(B.getBlockID());
  }  
  
  ExplodedNodeImpl* generateNodeImpl(void* State);
    
  CFGBlock* getBlock() const { return &B; }
};


template<typename STATE>
class GREndPathNodeBuilder  {
  typedef STATE                   StateTy;
  typedef ExplodedNode<StateTy>   NodeTy;
  
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
  
  StateTy* getState() const {
    return getPredecessor()->getState();
  }
  
  NodeTy* MakeNode(StateTy* St) {  
    return static_cast<NodeTy*>(NB.generateNodeImpl(St));
  }
};

  
template<typename SUBENGINE>
class GRCoreEngine : public GRCoreEngineImpl {
public:
  typedef SUBENGINE                              SubEngineTy; 
  typedef typename SubEngineTy::StateTy          StateTy;
  typedef ExplodedGraph<StateTy>                 GraphTy;
  typedef typename GraphTy::NodeTy               NodeTy;

protected:
  SubEngineTy& SubEngine;
  
  virtual void* getInitialState() {
    return SubEngine.getInitialState();
  }
  
  virtual void ProcessEndPath(GREndPathNodeBuilderImpl& BuilderImpl) {
    GREndPathNodeBuilder<StateTy> Builder(BuilderImpl);
    SubEngine.ProcessEndPath(Builder);
  }
  
  virtual void ProcessStmt(Stmt* S, GRStmtNodeBuilderImpl& BuilderImpl) {
    GRStmtNodeBuilder<StateTy> Builder(BuilderImpl);
    SubEngine.ProcessStmt(S, Builder);
  }
  
  virtual bool ProcessBlockEntrance(CFGBlock* Blk, void* State,
                                    GRBlockCounter BC) {    
    return SubEngine.ProcessBlockEntrance(Blk, static_cast<StateTy*>(State),BC);
  }

  virtual void ProcessBranch(Expr* Condition, Stmt* Terminator,
                             GRBranchNodeBuilderImpl& BuilderImpl) {
    GRBranchNodeBuilder<StateTy> Builder(BuilderImpl);
    SubEngine.ProcessBranch(Condition, Terminator, Builder);    
  }
  
  virtual void ProcessIndirectGoto(GRIndirectGotoNodeBuilderImpl& BuilderImpl) {
    GRIndirectGotoNodeBuilder<StateTy> Builder(BuilderImpl);
    SubEngine.ProcessIndirectGoto(Builder);
  }
  
  virtual void ProcessSwitch(GRSwitchNodeBuilderImpl& BuilderImpl) {
    GRSwitchNodeBuilder<StateTy> Builder(BuilderImpl);
    SubEngine.ProcessSwitch(Builder);
  }
  
public:  
  /// Construct a GRCoreEngine object to analyze the provided CFG using
  ///  a DFS exploration of the exploded graph.
  GRCoreEngine(CFG& cfg, Decl& cd, ASTContext& ctx, SubEngineTy& subengine)
    : GRCoreEngineImpl(new GraphTy(cfg, cd, ctx), GRWorkList::MakeDFS()),
      SubEngine(subengine) {}
  
  /// Construct a GRCoreEngine object to analyze the provided CFG and to
  ///  use the provided worklist object to execute the worklist algorithm.
  ///  The GRCoreEngine object assumes ownership of 'wlist'.
  GRCoreEngine(CFG& cfg, Decl& cd, ASTContext& ctx, GRWorkList* wlist,
               SubEngineTy& subengine)
    : GRCoreEngineImpl(new GraphTy(cfg, cd, ctx), wlist),
      SubEngine(subengine) {}
  
  virtual ~GRCoreEngine() {}
  
  /// getGraph - Returns the exploded graph.
  GraphTy& getGraph() {
    return *static_cast<GraphTy*>(G.get());
  }
  
  /// takeGraph - Returns the exploded graph.  Ownership of the graph is
  ///  transfered to the caller.
  GraphTy* takeGraph() { 
    return static_cast<GraphTy*>(G.take());
  }
};

} // end clang namespace

#endif
