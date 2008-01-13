//=-- ExplodedNode.h - Local, Path-Sensitive Supergraph Vertices -*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the template class ExplodedNode which is used to
//  represent a node in the location*state "exploded graph" of an
//  intra-procedural, path-sensitive dataflow analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_EXPLODEDNODE
#define LLVM_CLANG_ANALYSIS_EXPLODEDNODE

#include "clang/Analysis/ProgramPoint.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include <vector>

namespace clang {

class ExplodedGraphImpl;
class ExplodedNodeImpl;

/// ExplodeNodeGroup - A utility class used to represent the set of successor
///  and predecessor nodes of a node.  Most nodes will have only 1 successor
///  and 1 predecessor, so class allows us to store such unit sets of nodes
///  using a single pointer without allocating an entire vector.  For
///  larger sets of nodes, we dynamically allocate a vector.  This class
///  will likely be revised in the future to further improve performance and
///  to reduce memory footprint.
class ExplodedNodeGroup {
  enum { Size1 = 0x0, SizeOther = 0x1, Flags = 0x1 };
  uintptr_t P;

  unsigned getKind() const { return P & Flags; }

  std::vector<ExplodeNodeImpl*>& getVector() {
    assert (getKind() == SizeOther);
    return *reinterpret_cast<std::vector<ExplodeNodeImpl*>*>(P & ~Flags);
  }

  ExplodeNodeImpl* getNode() {
    assert (getKind() == Size1);
    return reinterpret_cast<ExplodeNodeImpl*>(P);
  }

public:
  ExplodedNodeGroup() : P(0) {}

  ~ExplodedNodeGroup() { if (getKind() == SizeOther) delete &getVector(); }

  inline ExplodedNodeImpl** begin() const {
    if (getKind() == Size1)
      return (ExplodedNodeImpl**) &P;
    else
      return getVector().begin();
  }
  
  inline ExplodedNodeImpl** end() const {
    if (getKind() == Size1)
      return ((ExplodedNodeImpl**) &P)+1;
    else
      return getVector().end();
  }
    
  inline unsigned size() const {
    if (getKind() == Size1)
      return getNode() ? 1 : 0;
    else
      return getVector().size();
  }
  
  inline bool empty() const {
    if (getKind() == Size1)
      return getNode() ? false : true;
    else
      return getVector().empty();
  }
  
  inline void addNode(ExplodedNodeImpl* N) {
    if (getKind() == Size1) {
      if (ExplodedNodeImpl* NOld = getNode()) {
        std::vector<ExplodeNodeImpl*>* V = new std::vector<ExplodeNodeImpl*>();
        V->push_back(NOld);
        V->push_back(N);
        P = reinterpret_cast<uintptr_t>(V) & SizeOther;
      }
      else
        P = reinterpret_cast<uintptr_t>(N);
    }
    else
      getVector().push_back(N);
  }
};

/// ExplodeNodeImpl - 
class ExplodedNodeImpl : public llvm::FoldingSetNode {
protected:
  friend class ExplodedGraphImpl;
    
  /// Location - The program location (within a function body) associated
  ///  with this node.
  const ProgramPoint Location;
  
  /// State - The state associated with this node. Normally this value
  ///  is immutable, but we anticipate there will be times when algorithms
  ///  that directly manipulate the analysis graph will need to change it.
  void* State;

  /// Preds - The predecessors of this node.
  ExplodedNodeGroup Preds;
  
  /// Succs - The successors of this node.
  ExplodedNodeGroup Succs;

  /// Construct a ExplodedNodeImpl with the provided location and state.
  explicit ExplodedNodeImpl(const ProgramLocation& loc, void* state)
    : Location(loc), State(state) {}

  /// addPredeccessor - Adds a predecessor to the current node, and 
  ///  in tandem add this node as a successor of the other node.  This
  ///  method is intended to be used only by ExplodedGraphImpl.
  void addPredecessor(ExplodedNodeImpl* V) {
    Preds.addNode(V);
    V->Succs.addNode(this);
  }

public:
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
  explicit ExplodedNode(unsigned ID, const ProgramEdge& loc, StateTy state)
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
  typedef const ExplodedNode** const_pred_pred_iterator;
                     
  pred_iterator pred_begin() { return (ExplodedNode**) Pred.begin(); }  
  pred_iterator pred_end() { return (ExplodedNode**) Pred.end(); }

  const_pred_iterator pred_begin() const {
    return const_cast<ExplodedNode*>(this)->pred_begin();
  }  
  const_pred_iterator pred_end() const {
    return const_cast<ExplodedNode*>(this)->pred_end();
  }
  
  succ_iterator succ_begin() { return (ExplodedNode**) Succ.begin(); }
  succ_iterator succ_end() { return (ExplodedNode**) Succ.end(); }
  
  const_succ_iterator succ_begin() const {
    return const_cast<ExplodedNode*>(this)->succ_begin();
  }  
  const_succ_iterator succ_end() const {
    return const_cast<ExplodedNode*>(this)->succ_end();
  }  
};
  
} // end clang namespace

// GraphTraits for ExplodedNodes.

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
