//===- DataStructureGraph.h - Provide graph classes --------------*- C++ -*--=//
//
// This file provides GraphTraits specializations for the DataStructure graph
// nodes, allowing datastructure graphs to be processed by generic graph
// algorithms.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DATASTRUCTURE_GRAPH_H
#define LLVM_ANALYSIS_DATASTRUCTURE_GRAPH_H

#include "Support/GraphTraits.h"
#include "llvm/Analysis/DataStructure.h"

class DSNodeIterator : public std::forward_iterator<DSNode, ptrdiff_t> {
  friend class DSNode;
  DSNode * const Node;
  unsigned Link;
  unsigned LinkIdx;
  
  typedef DSNodeIterator _Self;

  DSNodeIterator(DSNode *N) : Node(N), Link(0), LinkIdx(0) {   // begin iterator
    unsigned NumLinks = Node->getNumOutgoingLinks();
    while (Link < NumLinks && Node->getOutgoingLink(Link).empty())
      ++Link;
  }
  DSNodeIterator(DSNode *N, bool)       // Create end iterator
    : Node(N), Link(N->getNumOutgoingLinks()), LinkIdx(0) {
  }
public:

  bool operator==(const _Self& x) const {
    return Link == x.Link && LinkIdx == x.LinkIdx;
  }
  bool operator!=(const _Self& x) const { return !operator==(x); }
  
  pointer operator*() const {
    return Node->getOutgoingLink(Link)[LinkIdx].getNode();
  }
  pointer operator->() const { return operator*(); }
  
  _Self& operator++() {                // Preincrement
    if (LinkIdx < Node->getOutgoingLink(Link).size()-1)
      ++LinkIdx;
    else {
      unsigned NumLinks = Node->getNumOutgoingLinks();
      do {
        ++Link;
      } while (Link < NumLinks && Node->getOutgoingLink(Link).empty());
      LinkIdx = 0;
    }
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
