//===- DSGraphTraits.h - Provide generic graph interface --------*- C++ -*-===//
//
// This file provides GraphTraits specializations for the DataStructure graph
// nodes, allowing datastructure graphs to be processed by generic graph
// algorithms.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DSGRAPHTRAITS_H
#define LLVM_ANALYSIS_DSGRAPHTRAITS_H

#include "llvm/Analysis/DSGraph.h"
#include "Support/GraphTraits.h"
#include "Support/iterator"
#include "Support/STLExtras.h"

class DSNodeIterator : public forward_iterator<DSNode, ptrdiff_t> {
  friend class DSNode;
  DSNode * const Node;
  unsigned Offset;
  
  typedef DSNodeIterator _Self;

  DSNodeIterator(DSNode *N) : Node(N), Offset(0) {}   // begin iterator
  DSNodeIterator(DSNode *N, bool)       // Create end iterator
    : Node(N), Offset(N->getSize()) {
  }
public:
  DSNodeIterator(const DSNodeHandle &NH)
    : Node(NH.getNode()), Offset(NH.getOffset()) {}

  bool operator==(const _Self& x) const {
    return Offset == x.Offset;
  }
  bool operator!=(const _Self& x) const { return !operator==(x); }

  const _Self &operator=(const _Self &I) {
    assert(I.Node == Node && "Cannot assign iterators to two different nodes!");
    Offset = I.Offset;
    return *this;
  }
  
  pointer operator*() const {
    DSNodeHandle *NH = Node->getLink(Offset);
    return NH ? NH->getNode() : 0;
  }
  pointer operator->() const { return operator*(); }
  
  _Self& operator++() {                // Preincrement
    ++Offset;
    return *this;
  }
  _Self operator++(int) { // Postincrement
    _Self tmp = *this; ++*this; return tmp; 
  }

  unsigned getOffset() const { return Offset; }
  DSNode *getNode() const { return Node; }
};

// Provide iterators for DSNode...
inline DSNode::iterator DSNode::begin() { return DSNodeIterator(this); }
inline DSNode::iterator DSNode::end()   { return DSNodeIterator(this, false); }

template <> struct GraphTraits<DSNode*> {
  typedef DSNode NodeType;
  typedef DSNode::iterator ChildIteratorType;

  static NodeType *getEntryNode(NodeType *N) { return N; }
  static ChildIteratorType child_begin(NodeType *N) { return N->begin(); }
  static ChildIteratorType child_end(NodeType *N) { return N->end(); }
};

static DSNode &dereference(DSNode *N) { return *N; }

template <> struct GraphTraits<DSGraph*> {
  typedef DSNode NodeType;
  typedef DSNode::iterator ChildIteratorType;

  typedef std::pointer_to_unary_function<DSNode *, DSNode&> DerefFun;

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  typedef mapped_iterator<std::vector<DSNode*>::iterator,
                          DerefFun> nodes_iterator;
  static nodes_iterator nodes_begin(DSGraph *G) { return map_iterator(G->getNodes().begin(), DerefFun(dereference));}
  static nodes_iterator nodes_end  (DSGraph *G) { return map_iterator(G->getNodes().end(), DerefFun(dereference)); }

  static ChildIteratorType child_begin(NodeType *N) { return N->begin(); }
  static ChildIteratorType child_end(NodeType *N) { return N->end(); }
};

#endif
