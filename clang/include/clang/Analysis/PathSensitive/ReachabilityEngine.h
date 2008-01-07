//==- ReachabilityEngine.h - Path-Sens. Dataflow Engine ------------*- C++ -*-//
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

#ifndef LLVM_CLANG_ANALYSIS_REACHABILITYENGINE
#define LLVM_CLANG_ANALYSIS_REACHABILITYENGINE

#include "clang/AST/CFG.h"
#include "clang/Analysis/PathSensitive/ExplodedGraph.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/OwningPtr.h"

namespace clang {

namespace reng {
  class WorkList {
  public:
    virtual ~WorkList();
    virtual bool hasWork() const = 0;
    virtual void Enqueue(ExplodedNodeImpl* V) = 0;
    virtual ExplodedNodeImpl* Dequeue() = 0;
  };
  
  class DFS : public WorkList {
    llvm::SmallVector<ExplodedNodeImpl*,20> Stack;
  public:
    virtual ~DFS();

    virtual bool hasWork() const {
      return !Stack.empty();
    }
      
    virtual void Enqueue(ExplodedNodeImpl* V) {
      Stack.push_back(V);
    }
      
    virtual ExplodedNodeImpl* Dequeue() {
      ExplodedNodeImpl* V = Stack.back();
      Stack.pop_back();
      return V;
    }      
  };
}

class ReachabilityEngineImpl {
protected:  
  typedef llvm::DenseMap<Stmt*,Stmt*> ParentMapTy;

  /// cfg - The control-flow graph of the function being analyzed.
  CFG& cfg;
    
  /// G - The simulation graph.  Each node is a (location,state) pair.
  llvm::OwningPtr<ExplodedGraphImpl> G;
  
  /// ParentMap - A lazily populated map from a Stmt* to its parent Stmt*.
  ParentMapTy ParentMap;
  
  /// CurrentBlkExpr - The current Block-level expression being processed.
  ///  This is used when lazily populating ParentMap.
  Stmt* CurrentBlkExpr;
  
  /// WList - A set of queued nodes that need to be processed by the
  ///  worklist algorithm.  It is up to the implementation of WList to decide
  ///  the order that nodes are processed.
  llvm::OwningPtr<reng::WorkList> WList;  
  
  //==----------------------------------------------------------------------==//
  // Internal methods. 
  
  /// getNode - Implemented by ReachabilityEngine<> subclass. 
  ///   Creates/fetches a node and inserts it into the 
  ExplodedNodeImpl* getNode(const ProgramEdge& Loc, void* State,
                            ExplodedNodeImpl* Pred);
  
  inline ExplodedNodeImpl* getNode(const ProgramEdge& Loc,
                                   ExplodedNodeImpl* Pred) {
    
    return getNode(Loc,Pred->State,Pred);
  }
  
  /// getInitialState - Gets the void* representing the initial 'state'
  ///  of the analysis.  This is simply a wrapper (implemented
  ///  in ReachabilityEngine) that performs type erasure on the initial
  ///  state returned by the checker object.
  virtual void* getInitialState() = 0;
  
  /// PopulateParentMap - Populates ParentMap starting from the specified
  ///   expression.
  void PopulateParentMap(Stmt* Parent);  
  
  void ProcessBlkBlk(const BlkBlkEdge& E, ExplodedNodeImpl* Pred);
  void ProcessBlkStmt(const BlkStmtEdge& E, ExplodedNodeImpl* Pred);
  void ProcessStmtBlk(const StmtBlkEdge& E, ExplodedNodeImpl* Pred);

  virtual void ProcessEOP(CFGBlock* Blk, ExplodedNodeImpl* Pred);
  virtual void ProcessStmt(Stmt* S, ExplodedNodeImpl* Pred);
  virtual void ProcessTerminator(Stmt* Terminator, ExplodedNodeImpl* Pred);

private:
  ReachabilityEngineImpl(const ReachabilityEngineImpl&); // Do not implement.
  ReachabilityEngineImpl& operator=(const ReachabilityEngineImpl&);
  
protected:  
  ReachabilityEngineImpl(CFG& c, reng::WorkList* wl);
  
public:
  /// ExecuteWorkList - Run the worklist algorithm for a maximum number of
  ///  steps.  Returns true if there is still simulation state on the worklist.
  bool ExecuteWorkList(unsigned Steps = 1000000);
  
  virtual ~ReachabilityEngineImpl() {}
};
  
  
template<typename CHECKER>
class ReachabilityEngine : public ReachabilityEngineImpl {
public:
  typedef CHECKER                                CheckerTy; 
  typedef typename CheckerTy::StateTy            StateTy;
  typedef ExplodedGraph<CheckerTy>               GraphTy;
  typedef typename GraphTy::NodeTy               NodeTy;

protected:

  virtual void* getInitialState() {
    return (void*) getCheckerState()->getInitialState();        
  }
  
  virtual void ProcessEOP(CFGBlock* Blk, ExplodedNodeImpl* Pred) {
    assert (false && "Not implemented yet.");
    // FIXME: Perform dispatch to adjust state.
//    ExplodedNodeImpl* V = G->getNodeImpl(BlkStmtEdge(Blk,NULL),
//                                         Pred->State).first;
    
//    V->addPredecessor(Pred);
//    Graph.addEndOfPath(V);  
  }
    
  
  virtual void ProcessStmt(Stmt* S, ExplodedNodeImpl* Pred) {
    CurrentBlkExpr = S;    
    assert(false && "Not implemented.");    
    CurrentBlkExpr = NULL;
  }
    
  virtual void ProcessTerminator(Stmt* Terminator, ExplodedNodeImpl* Pred) {
    assert(false && "Not implemented.");
  }
  
  
public:  
  /// Construct a ReachabilityEngine object to analyze the provided CFG using
  ///  a DFS exploration of the exploded graph.
  ReachabilityEngine(CFG& Cfg)
    : ReachabilityEngineImpl(cfg,new reng::DFS()) {}
  
  /// Construct a ReachabilityEngine object to analyze the provided CFG and to
  ///  use the provided worklist object to execute the worklist algorithm.
  ///  The ReachabilityEngine object assumes ownership of 'wlist'.
  ReachabilityEngine(CFG& cfg, reng::WorkList* wlist) 
    : ReachabilityEngineImpl(cfg,wlist) {}
  
  /// getGraph - Returns the exploded graph.  Ownership of the graph remains
  ///  with the ReachabilityEngine object.
  GraphTy* getGraph() const { return static_cast<GraphTy*>(G.get()); }
  
  /// getCheckerState - Returns the internal checker state.  Ownership is not
  ///  transferred to the caller.
  CheckerTy* getCheckerState() const {
    return static_cast<GraphTy*>(G.get())->getCheckerState();
  }  
  
  /// takeGraph - Returns the exploded graph.  Ownership of the graph is
  ///  transfered to the caller.
  GraphTy* takeGraph() { return static_cast<GraphTy*>(G.take()); }
};

} // end clang namespace

#endif
