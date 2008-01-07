//===-- ExplodedGraph.h - Local, Path-Sensitive Supergraph -*- C++ -*------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the template class ExplodedGraph, which represents a
//  path-sensitive, intra-procedural dataflow "exploded graph."
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_EXPLODEDGRAPH
#define LLVM_CLANG_ANALYSIS_EXPLODEDGRAPH

#include "clang/Analysis/PathSensitive/ExplodedNode.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Allocator.h"

namespace clang {
  
class ReachabilityEngineImpl;
  
class ExplodedGraphImpl {
protected:
  friend class ReachabilityEngineImpl;
  
  // Type definitions.
  typedef llvm::DenseMap<ProgramEdge,void*>         EdgeNodeSetMap;  
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
  ///  is intended to be used only by ReachabilityEngineImpl.
  virtual ExplodedNodeImpl* getNodeImpl(const ProgramEdge& L, void* State,
                                       bool& IsNew) = 0;
                                                            
  /// addRoot - Add an untyped node to the set of roots.
  void addRoot(ExplodedNodeImpl* V) { Roots.push_back(V); }

  /// addEndOfPath - Add an untyped node to the set of EOP nodes.
  void addEndOfPath(ExplodedNodeImpl* V) { EndNodes.push_back(V); }

public:
  virtual ~ExplodedGraphImpl() {};
  
  unsigned num_roots() const { return Roots.size(); }
  unsigned num_eops() const { return EndNodes.size(); }  
  unsigned getCounter() const { return NodeCounter; }
};
  
template <typename STATE>
class ExplodedGraph : public ExplodedGraphImpl {
public:
  typedef STATE                  StateTy;
  typedef ExplodedNode<StateTy>  NodeTy;
  
protected:
  virtual ExplodedNodeImpl*
  getNodeImpl(const ProgramEdge& L, void* State, bool& IsNew) {
    return getNode(L,ReachabilityTrait<StateTy>::toState(State),&IsNew);
  }
    
public:
  virtual ~ExplodedGraph() {
    // Delete the FoldingSet's in Nodes.  Note that the contents
    // of the FoldingSets are nodes allocated from the BumpPtrAllocator,
    // so all of those will get nuked when that object is destroyed.
    for (EdgeNodeSetMap::iterator I=Nodes.begin(), E=Nodes.end(); I!=E; ++I)
      delete reinterpret_cast<llvm::FoldingSet<NodeTy>*>(I->second);
  }
  
  /// getNode - Retrieve the node associated with a (Location,State) pair,
  ///  where the 'Location' is a ProgramEdge in the CFG.  If no node for
  ///  this pair exists, it is created.  IsNew is set to true if
  ///  the node was freshly created.
  NodeTy* getNode(const ProgramEdge& L, StateTy State, bool* IsNew = NULL) {
    
    // Retrieve the node set associated with Loc.
    llvm::FoldingSet<NodeTy>*& VSet =
       reinterpret_cast<llvm::FoldingSet<NodeTy>*&>(Nodes[L]);
    
    // Create the FoldingSet for the nodes if it does not exist yet.
    if (!VSet) VSet = new llvm::FoldingSet<NodeTy>();
    
    // Profile 'State' to determine if we already have an existing node.
    llvm::FoldingSetNodeID profile;    
    void* InsertPos = 0;
    
    StateTy::Profile(profile, State);
    NodeTy* V = VSet.FindNodeOrInsertPos(profile,InsertPos);

    if (!V) {
      // Allocate a new node.
      V = (NodeTy*) Allocator.Allocate<NodeTy>();
      new (V) NodeTy(NodeCounter++,L,State);
      
      // Insert the node into the node set and return it.
      VSet.InsertNode(V,InsertPos);
      
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

#endif
