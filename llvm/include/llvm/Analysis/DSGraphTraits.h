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
#include "llvm/Value.h"  // FIXME: Move cast/dyn_cast out to Support

class DSNodeIterator : public std::forward_iterator<DSNode, ptrdiff_t> {
  DSNode * const Node;
  unsigned Link;
  unsigned LinkIdx;
  
  typedef DSNodeIterator _Self;
public:
  DSNodeIterator(DSNode *N) : Node(N), Link(0), LinkIdx(0) {}  // begin iterator
  DSNodeIterator(DSNode *N, bool)       // Create end iterator
    : Node(N), Link(N->getNumOutgoingLinks()), LinkIdx(0) {
  }

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
      ++Link;
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
  typedef DSNodeIterator ChildIteratorType;

  static NodeType *getEntryNode(DSNode *N) { return N; }
  static ChildIteratorType child_begin(NodeType *N) { 
    return DSNodeIterator(N);
  }
  static ChildIteratorType child_end(NodeType *N) { 
    return DSNodeIterator(N, true);
  }
};


#endif
