//==- GREngine.h - Path-Sensitive Dataflow Engine ------------------*- C++ -*-//
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

#ifndef LLVM_CLANG_ANALYSIS_GRENGINE
#define LLVM_CLANG_ANALYSIS_GRENGINE

#include "clang/Analysis/PathSensitive/ExplodedGraph.h"
#include "clang/Analysis/PathSensitive/GRWorkList.h"
#include "llvm/ADT/OwningPtr.h"

namespace clang {
  
class CFG;
class GRNodeBuilderImpl;
class GRWorkList;
  
class GREngineImpl {
protected:
  friend class GRNodeBuilderImpl;
  
  typedef llvm::DenseMap<Stmt*,Stmt*> ParentMapTy;

  /// cfg - The control-flow graph of the function being analyzed.
  CFG& cfg;
    
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
  
  void GenerateNode(const ProgramPoint& Loc, void* State, 
                    ExplodedNodeImpl* Pred = NULL);
  
  /// getInitialState - Gets the void* representing the initial 'state'
  ///  of the analysis.  This is simply a wrapper (implemented
  ///  in GREngine) that performs type erasure on the initial
  ///  state returned by the checker object.
  virtual void* getInitialState() = 0;
  
  void HandleBlockEdge(const BlockEdge& E, ExplodedNodeImpl* Pred);
  void HandleBlockEntrance(const BlockEntrance& E, ExplodedNodeImpl* Pred);
  void HandleBlockExit(CFGBlock* B, ExplodedNodeImpl* Pred);
  void HandlePostStmt(const PostStmt& S, CFGBlock* B,
                      unsigned StmtIdx, ExplodedNodeImpl *Pred);

  virtual void* ProcessEOP(CFGBlock* Blk, void* State) = 0;  

  virtual void ProcessStmt(Stmt* S, GRNodeBuilderImpl& Builder) = 0;

  virtual void ProcessTerminator(Stmt* Terminator, CFGBlock* B, 
                                 ExplodedNodeImpl* Pred) = 0;

private:
  GREngineImpl(const GREngineImpl&); // Do not implement.
  GREngineImpl& operator=(const GREngineImpl&);
  
protected:  
  GREngineImpl(CFG& c, ExplodedGraphImpl* g, GRWorkList* wl)
   : cfg(c), G(g), WList(wl) {}
  
public:
  /// ExecuteWorkList - Run the worklist algorithm for a maximum number of
  ///  steps.  Returns true if there is still simulation state on the worklist.
  bool ExecuteWorkList(unsigned Steps = 1000000);
  
  virtual ~GREngineImpl() {}
};
  
class GRNodeBuilderImpl {
  GREngineImpl& Eng;
  CFGBlock& B;
  const unsigned Idx;
  ExplodedNodeImpl* LastNode;  
  bool HasGeneratedNode;
  bool Populated;
  
  typedef llvm::SmallPtrSet<ExplodedNodeImpl*,5> DeferredTy;
  DeferredTy Deferred;
  
  void GenerateAutoTransition(ExplodedNodeImpl* N);
  
public:
  GRNodeBuilderImpl(CFGBlock* b, unsigned idx,
                    ExplodedNodeImpl* N, GREngineImpl* e);      
  
  ~GRNodeBuilderImpl();
  
  const ExplodedGraphImpl& getGraph() const { return *Eng.G; }

  inline ExplodedNodeImpl* getLastNode() {
    return LastNode ? (LastNode->isInfeasible() ? NULL : LastNode) : NULL;
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
};

template<typename CHECKER>
class GRNodeBuilder  {
  typedef CHECKER                                CheckerTy; 
  typedef typename CheckerTy::StateTy            StateTy;
  typedef ExplodedGraph<CheckerTy>               GraphTy;
  typedef typename GraphTy::NodeTy               NodeTy;
  
  GRNodeBuilderImpl& NB;
  
public:
  GRNodeBuilder(GRNodeBuilderImpl& nb) : NB(nb) {}
  
  const GraphTy& getGraph() const {
    return static_cast<const GraphTy&>(NB.getGraph());
  }
  
  NodeTy* getLastNode() const {
    return static_cast<NodeTy*>(NB.getLastNode());
  }
  
  NodeTy* generateNode(Stmt* S, StateTy State, NodeTy* Pred) {
    void *state = GRTrait<StateTy>::toPtr(State);        
    return static_cast<NodeTy*>(NB.generateNodeImpl(S, state, Pred));
  }
  
  NodeTy* generateNode(Stmt* S, StateTy State) {
    void *state = GRTrait<StateTy>::toPtr(State);
    return static_cast<NodeTy*>(NB.generateNodeImpl(S, state));    
  }  
};
  
  
template<typename CHECKER>
class GREngine : public GREngineImpl {
public:
  typedef CHECKER                                CheckerTy; 
  typedef typename CheckerTy::StateTy            StateTy;
  typedef ExplodedGraph<CheckerTy>               GraphTy;
  typedef typename GraphTy::NodeTy               NodeTy;

protected:
  // A local reference to the checker that avoids an indirect access
  // via the Graph.
  CheckerTy* Checker;
  
  
  virtual void* getInitialState() {
    return GRTrait<StateTy>::toPtr(getCheckerState().getInitialState());
  }
  
  virtual void* ProcessEOP(CFGBlock* Blk, void* State) {
    // FIXME: Perform dispatch to adjust state.
    return State;
  }
  
  virtual void ProcessStmt(Stmt* S, GRNodeBuilderImpl& BuilderImpl) {
    GRNodeBuilder<CHECKER> Builder(BuilderImpl);
    Checker->ProcessStmt(S, Builder);
  }

  virtual void ProcessTerminator(Stmt* Terminator, CFGBlock* B, 
                                 ExplodedNodeImpl* Pred) {
    // FIXME: Dispatch.    
  }
  
  
public:  
  /// Construct a GREngine object to analyze the provided CFG using
  ///  a DFS exploration of the exploded graph.
  GREngine(CFG& cfg)
  : GREngineImpl(cfg, new GraphTy(), GRWorkList::MakeDFS()),
      Checker(static_cast<GraphTy*>(G.get())->getCheckerState()) {
    Checker->Initialize(cfg);
  }
  
  /// Construct a GREngine object to analyze the provided CFG and to
  ///  use the provided worklist object to execute the worklist algorithm.
  ///  The GREngine object assumes ownership of 'wlist'.
  GREngine(CFG& cfg, GRWorkList* wlist) 
    : GREngineImpl(cfg, new GraphTy(), wlist),
      Checker(static_cast<GraphTy*>(G.get())->getCheckerState()) {
    Checker->Initialize(cfg);
  }
  
  virtual ~GREngine() {}
  
  /// getGraph - Returns the exploded graph.
  GraphTy& getGraph() {
    return *static_cast<GraphTy*>(G.get());
  }
  
  /// getCheckerState - Returns the internal checker state.
  CheckerTy& getCheckerState() {
    return *Checker;
  }  
  
  /// takeGraph - Returns the exploded graph.  Ownership of the graph is
  ///  transfered to the caller.
  GraphTy* takeGraph() { 
    return static_cast<GraphTy*>(G.take());
  }
};

} // end clang namespace

#endif
