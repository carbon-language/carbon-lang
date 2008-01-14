//=-- ExplodedGraph.h - Local, Path-Sens. "Exploded Graph" -*- C++ -*-------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the template classes ExplodedNode and ExplodedGraph,
//  which represent a path-sensitive, intra-procedural "exploded graph."
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_EXPLODEDGRAPH
#define LLVM_CLANG_ANALYSIS_EXPLODEDGRAPH

#include "clang/Analysis/ProgramPoint.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Allocator.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/DepthFirstIterator.h"

namespace clang {

class GREngineImpl;
class ExplodedNodeImpl;

class ExplodedNodeImpl : public llvm::FoldingSetNode {
protected:
  friend class ExplodedGraphImpl;
  
  class NodeGroup {
    enum { Size1 = 0x0, SizeOther = 0x1, Flags = 0x1 };
    uintptr_t P;
    
    unsigned getKind() const { return P & Flags; }
    void* getPtr() const { return reinterpret_cast<void*>(P & ~Flags); }
    ExplodedNodeImpl* getNode() const;
    
  public:
    NodeGroup() : P(0) {}
    
    ~NodeGroup();
    
    ExplodedNodeImpl** begin() const;
    
    ExplodedNodeImpl** end() const;
    
    unsigned size() const;
    
    bool empty() const;
    
    void addNode(ExplodedNodeImpl* N);
  };
  
  
  /// Location - The program location (within a function body) associated
  ///  with this node.
  const ProgramPoint Location;
  
  /// State - The state associated with this node. Normally this value
  ///  is immutable, but we anticipate there will be times when algorithms
  ///  that directly manipulate the analysis graph will need to change it.
  void* State;
  
  /// Preds - The predecessors of this node.
  NodeGroup Preds;
  
  /// Succs - The successors of this node.
  NodeGroup Succs;
  
  /// Construct a ExplodedNodeImpl with the provided location and state.
  explicit ExplodedNodeImpl(const ProgramPoint& loc, void* state)
  : Location(loc), State(state) {}
  
  /// addPredeccessor - Adds a predecessor to the current node, and 
  ///  in tandem add this node as a successor of the other node.  This
  ///  method is intended to be used only by ExplodedGraphImpl.
  void addPredecessor(ExplodedNodeImpl* V) {
    Preds.addNode(V);
    V->Succs.addNode(this);
  }
  
public:
  // This method is only defined so that we can cast a
  // void* to FoldingSet<ExplodedNodeImpl> so that we can iterate
  // over the vertices of EdgeNodeSetMap in ExplodeGraphImpl.
  // The actual profiling of vertices will be done in the derived
  // class, ExplodedNode<>.  Nodes will NEVER be INSERTED into the
  // FoldingSet using this Profile method (since it doesn't do anything).
  inline void Profile(llvm::FoldingSetNodeID& ID) const {
    assert (false && "Needs to be implemented in derived class.");
  }
  
  /// getLocation - Returns the edge associated with the given node.
  const ProgramPoint& getLocation() const { return Location; }
  
  unsigned succ_size() const { return Succs.size(); }
  unsigned pred_size() const { return Preds.size(); }
  bool succ_empty() const { return Succs.empty(); }
  bool pred_empty() const { return Preds.size(); }
};


template <typename StateTy>
struct GRTrait {
  static inline void* toPtr(StateTy S) {
    return reinterpret_cast<void*>(S);
  }  
  static inline StateTy toState(void* P) {
    return reinterpret_cast<StateTy>(P);
  }
};


template <typename StateTy>
class ExplodedNode : public ExplodedNodeImpl {
public:
  /// Construct a ExplodedNodeImpl with the given node ID, program edge,
  ///  and state.
  explicit ExplodedNode(unsigned ID, const ProgramPoint& loc, StateTy state)
  : ExplodedNodeImpl(ID, loc, GRTrait<StateTy>::toPtr(state)) {}
  
  /// getState - Returns the state associated with the node.  
  inline StateTy getState() const {
    return GRTrait<StateTy>::toState(State);
  }
  
  // Profiling (for FoldingSet).
  inline void Profile(llvm::FoldingSetNodeID& ID) const {
    StateTy::Profile(ID, getState());
  }
  
  // Iterators over successor and predecessor vertices.
  typedef ExplodedNode**       succ_iterator;
  typedef const ExplodedNode** const_succ_iterator;
  typedef ExplodedNode**       pred_iterator;
  typedef const ExplodedNode** const_pred_iterator;

  pred_iterator pred_begin() { return (ExplodedNode**) Preds.begin(); }
  pred_iterator pred_end() { return (ExplodedNode**) Preds.end(); }

  const_pred_iterator pred_begin() const {
    return const_cast<ExplodedNode*>(this)->pred_begin();
  }  
  const_pred_iterator pred_end() const {
    return const_cast<ExplodedNode*>(this)->pred_end();
  }

  succ_iterator succ_begin() { return (ExplodedNode**) Succs.begin(); }
  succ_iterator succ_end() { return (ExplodedNode**) Succs.end(); }

  const_succ_iterator succ_begin() const {
    return const_cast<ExplodedNode*>(this)->succ_begin();
  }
  const_succ_iterator succ_end() const {
    return const_cast<ExplodedNode*>(this)->succ_end();
  }
};


class ExplodedGraphImpl {
protected:
  friend class GREngineImpl;
  
  // Type definitions.
  typedef llvm::DenseMap<ProgramPoint,void*>        EdgeNodeSetMap;
  typedef llvm::SmallVector<ExplodedNodeImpl*,2>    RootsTy;
  typedef llvm::SmallVector<ExplodedNodeImpl*,10>   EndNodesTy;
  
  /// NodeCounter - The number of nodes that have been created, although
  ///  this need not be the current number of nodes in the graph that
  ///  are reachable from the roots.  This counter is used to assign a unique
  ///  number to each node (which is useful for debugging).
  unsigned NodeCounter;
  
  /// Roots - The roots of the simulation graph. Usually there will be only
  /// one, but clients are free to establish multiple subgraphs within a single
  /// SimulGraph. Moreover, these subgraphs can often merge when paths from
  /// different roots reach the same state at the same program location.
  RootsTy Roots;

  /// EndNodes - The nodes in the simulation graph which have been
  ///  specially marked as the endpoint of an abstract simulation path.
  EndNodesTy EndNodes;
    
  /// Nodes - A mapping from edges to nodes.
  EdgeNodeSetMap Nodes;
  
  /// Allocator - BumpPtrAllocator to create nodes.
  llvm::BumpPtrAllocator Allocator;

  /// getNodeImpl - Retrieve the node associated with a (Location,State)
  ///  pair, where 'State' is represented as an opaque void*.  This method
  ///  is intended to be used only by GREngineImpl.
  virtual ExplodedNodeImpl* getNodeImpl(const ProgramPoint& L, void* State,
                                        bool* IsNew) = 0;

  /// addRoot - Add an untyped node to the set of roots.
  ExplodedNodeImpl* addRoot(ExplodedNodeImpl* V) {
    Roots.push_back(V);
    return V;
  }

  /// addEndOfPath - Add an untyped node to the set of EOP nodes.
  ExplodedNodeImpl* addEndOfPath(ExplodedNodeImpl* V) {
    EndNodes.push_back(V);
    return V;
  }

public:
  virtual ~ExplodedGraphImpl();

  unsigned num_roots() const { return Roots.size(); }
  unsigned num_eops() const { return EndNodes.size(); }
  unsigned getCounter() const { return NodeCounter; }
};
  
template <typename CHECKER>
class ExplodedGraph : public ExplodedGraphImpl {
public:
  typedef CHECKER                     CheckerTy;
  typedef typename CHECKER::StateTy   StateTy;
  typedef ExplodedNode<StateTy>       NodeTy;
  
protected:
  llvm::OwningPtr<CheckerTy> CheckerState;

protected:
  virtual ExplodedNodeImpl*
  getNodeImpl(const ProgramPoint& L, void* State, bool* IsNew) {
    return getNode(L,GRTrait<StateTy>::toState(State),IsNew);
  }
    
public:
  /// getCheckerState - Returns the internal checker state associated
  ///  with the exploded graph.  Ownership remains with the ExplodedGraph
  ///  objecct.
  CheckerTy* getCheckerState() const { return CheckerState.get(); }
  
  /// getNode - Retrieve the node associated with a (Location,State) pair,
  ///  where the 'Location' is a ProgramPoint in the CFG.  If no node for
  ///  this pair exists, it is created.  IsNew is set to true if
  ///  the node was freshly created.
  NodeTy* getNode(const ProgramPoint& L, StateTy State, bool* IsNew = NULL) {
    
    // Retrieve the node set associated with Loc.
    llvm::FoldingSet<NodeTy>*& VSet =
       reinterpret_cast<llvm::FoldingSet<NodeTy>*&>(Nodes[L]);
    
    // Create the FoldingSet for the nodes if it does not exist yet.
    if (!VSet) VSet = new llvm::FoldingSet<NodeTy>();
    
    // Profile 'State' to determine if we already have an existing node.
    llvm::FoldingSetNodeID profile;    
    void* InsertPos = 0;
    
    StateTy::Profile(profile, State);
    NodeTy* V = VSet.FindNodeOrInsertPos(profile, InsertPos);

    if (!V) {
      // Allocate a new node.
      V = (NodeTy*) Allocator.Allocate<NodeTy>();
      new (V) NodeTy(NodeCounter++, L, State);
      
      // Insert the node into the node set and return it.
      VSet.InsertNode(V, InsertPos);
      
      if (IsNew) *IsNew = true;
    }
    else
      if (IsNew) *IsNew = false;

    return V;
  }
  
  // Iterators.
  typedef NodeTy*         roots_iterator;
  typedef const NodeTy*   const_roots_iterator;
  typedef NodeTy*         eop_iterator;
  typedef const NodeTy*   const_eop_iterator;
  
  
  roots_iterator roots_begin() {
    return static_cast<NodeTy*>(Roots.begin());
  }
  
  roots_iterator roots_end() { 
    return static_cast<NodeTy*>(Roots.end());
  }
  
  const_roots_iterator roots_begin() const { 
    return const_cast<ExplodedGraph>(this)->roots_begin();
  }
  
  const_roots_iterator roots_end() const { 
    return const_cast<ExplodedGraph>(this)->roots_end();
  }  

  eop_iterator eop_begin() {
    return static_cast<NodeTy*>(EndNodes.begin());
  }
    
  eop_iterator eop_end() { 
    return static_cast<NodeTy*>(EndNodes.end());
  }
  
  const_eop_iterator eop_begin() const {
    return const_cast<ExplodedGraph>(this)->eop_begin();
  }
  
  const_eop_iterator eop_end() const {
    return const_cast<ExplodedGraph>(this)->eop_end();
  }
};
  
} // end clang namespace

// GraphTraits

namespace llvm {
  template<typename StateTy>
  struct GraphTraits<clang::ExplodedNode<StateTy>*> {
    typedef clang::ExplodedNode<StateTy>      NodeType;
    typedef typename NodeType::succ_iterator  ChildIteratorType;
    typedef llvm::df_iterator<NodeType*>      nodes_iterator;
    
    static inline NodeType* getEntryNode(NodeType* N) {
      return N;
    }
    
    static inline ChildIteratorType child_begin(NodeType* N) {
      return N->succ_begin();
    }
    
    static inline ChildIteratorType child_end(NodeType* N) {
      return N->succ_end();
    }
    
    static inline nodes_iterator nodes_begin(NodeType* N) {
      return df_begin(N);
    }
    
    static inline nodes_iterator nodes_end(NodeType* N) {
      return df_end(N);
    }
  };
  
  template<typename StateTy>
  struct GraphTraits<const clang::ExplodedNode<StateTy>*> {
    typedef const clang::ExplodedNode<StateTy> NodeType;
    typedef typename NodeType::succ_iterator   ChildIteratorType;
    typedef llvm::df_iterator<NodeType*>       nodes_iterator;
    
    static inline NodeType* getEntryNode(NodeType* N) {
      return N;
    }
    
    static inline ChildIteratorType child_begin(NodeType* N) {
      return N->succ_begin();
    }
    
    static inline ChildIteratorType child_end(NodeType* N) {
      return N->succ_end();
    }
    
    static inline nodes_iterator nodes_begin(NodeType* N) {
      return df_begin(N);
    }
    
    static inline nodes_iterator nodes_end(NodeType* N) {
      return df_end(N);
    }
  };
  
} // end llvm namespace

#endif
