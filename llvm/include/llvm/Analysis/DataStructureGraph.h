//===- DataStructureGraph.h - Provide graph classes --------------*- C++ -*--=//
//
// This file provides GraphTraits specializations for the DataStructure graph
// nodes, allowing datastructure graphs to be processed by generic graph
// algorithms.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DATASTRUCTURE_GRAPH_H
#define LLVM_ANALYSIS_DATASTRUCTURE_GRAPH_H

#include "llvm/Analysis/DataStructure.h"
#include "Support/GraphTraits.h"
#include "Support/iterator"

#if 0

class DSNodeIterator : public forward_iterator<DSNode, ptrdiff_t> {
  friend class DSNode;
  DSNode * const Node;
  unsigned Link;
  
  typedef DSNodeIterator _Self;

  DSNodeIterator(DSNode *N) : Node(N), Link(0) {   // begin iterator
    unsigned NumLinks = Node->getNumLinks();
    while (Link < NumLinks && Node->getLink(Link) == 0)
      ++Link;
  }
  DSNodeIterator(DSNode *N, bool)       // Create end iterator
    : Node(N), Link(N->getNumLinks()) {
  }
public:

  bool operator==(const _Self& x) const {
    return Link == x.Link;
  }
  bool operator!=(const _Self& x) const { return !operator==(x); }
  
  pointer operator*() const {
    return Node->getLink(Link);
  }
  pointer operator->() const { return operator*(); }
  
  _Self& operator++() {                // Preincrement
    unsigned NumLinks = Node->getNumLinks();
    do {
      ++Link;
    } while (Link < NumLinks && Node->getLink(Link) != 0);
    return *this;
  }
  _Self operator++(int) { // Postincrement
    _Self tmp = *this; ++*this; return tmp; 
  }
};


template <> struct GraphTraits<DSNode*> {
  typedef DSNode NodeType;
  typedef DSNode::iterator ChildIteratorType;

  static NodeType *getEntryNode(DSNode *N) { return N; }
  static ChildIteratorType child_begin(NodeType *N) { return N->begin(); }
  static ChildIteratorType child_end(NodeType *N) { return N->end(); }
};

// Provide iterators for DSNode...
inline DSNode::iterator DSNode::begin() { return DSNodeIterator(this); }
inline DSNode::iterator DSNode::end()   { return DSNodeIterator(this, false); }

#endif

#endif
