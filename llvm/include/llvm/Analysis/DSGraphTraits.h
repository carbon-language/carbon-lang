//===- DSGraphTraits.h - Provide generic graph interface --------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
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

namespace llvm {

template<typename NodeTy>
class DSNodeIterator : public forward_iterator<const DSNode, ptrdiff_t> {
  friend class DSNode;
  NodeTy * const Node;
  unsigned Offset;
  
  typedef DSNodeIterator<NodeTy> _Self;

  DSNodeIterator(NodeTy *N) : Node(N), Offset(0) {}   // begin iterator
  DSNodeIterator(NodeTy *N, bool) : Node(N) {         // Create end iterator
    Offset = N->getNumLinks() << DS::PointerShift;
    if (Offset == 0 && Node->getForwardNode() &&
        Node->isDeadNode())        // Model Forward link
      Offset += DS::PointerSize;
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
    if (Node->isDeadNode())
      return Node->getForwardNode();
    else
      return Node->getLink(Offset).getNode();
  }
  pointer operator->() const { return operator*(); }
  
  _Self& operator++() {                // Preincrement
    Offset += (1 << DS::PointerShift);
    return *this;
  }
  _Self operator++(int) { // Postincrement
    _Self tmp = *this; ++*this; return tmp; 
  }

  unsigned getOffset() const { return Offset; }
  const DSNode *getNode() const { return Node; }
};

// Provide iterators for DSNode...
inline DSNode::iterator DSNode::begin() {
  return DSNode::iterator(this);
}
inline DSNode::iterator DSNode::end() {
  return DSNode::iterator(this, false);
}
inline DSNode::const_iterator DSNode::begin() const {
  return DSNode::const_iterator(this);
}
inline DSNode::const_iterator DSNode::end() const {
  return DSNode::const_iterator(this, false);
}

template <> struct GraphTraits<DSNode*> {
  typedef DSNode NodeType;
  typedef DSNode::iterator ChildIteratorType;

  static NodeType *getEntryNode(NodeType *N) { return N; }
  static ChildIteratorType child_begin(NodeType *N) { return N->begin(); }
  static ChildIteratorType child_end(NodeType *N) { return N->end(); }
};

template <> struct GraphTraits<const DSNode*> {
  typedef const DSNode NodeType;
  typedef DSNode::const_iterator ChildIteratorType;

  static NodeType *getEntryNode(NodeType *N) { return N; }
  static ChildIteratorType child_begin(NodeType *N) { return N->begin(); }
  static ChildIteratorType child_end(NodeType *N) { return N->end(); }
};

static       DSNode &dereference (      DSNode *N) { return *N; }
static const DSNode &dereferenceC(const DSNode *N) { return *N; }

template <> struct GraphTraits<DSGraph*> {
  typedef DSNode NodeType;
  typedef DSNode::iterator ChildIteratorType;

  typedef std::pointer_to_unary_function<DSNode *, DSNode&> DerefFun;

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  typedef mapped_iterator<std::vector<DSNode*>::iterator,
                          DerefFun> nodes_iterator;
  static nodes_iterator nodes_begin(DSGraph *G) {
    return map_iterator(G->getNodes().begin(), DerefFun(dereference));
  }
  static nodes_iterator nodes_end(DSGraph *G) {
    return map_iterator(G->getNodes().end(), DerefFun(dereference));
  }

  static ChildIteratorType child_begin(NodeType *N) { return N->begin(); }
  static ChildIteratorType child_end(NodeType *N) { return N->end(); }
};

template <> struct GraphTraits<const DSGraph*> {
  typedef const DSNode NodeType;
  typedef DSNode::const_iterator ChildIteratorType;

  typedef std::pointer_to_unary_function<const DSNode *,const DSNode&> DerefFun;

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  typedef mapped_iterator<std::vector<DSNode*>::const_iterator,
                          DerefFun> nodes_iterator;
  static nodes_iterator nodes_begin(const DSGraph *G) {
    return map_iterator(G->getNodes().begin(), DerefFun(dereferenceC));
  }
  static nodes_iterator nodes_end(const DSGraph *G) {
    return map_iterator(G->getNodes().end(), DerefFun(dereferenceC));
  }

  static ChildIteratorType child_begin(const NodeType *N) { return N->begin(); }
  static ChildIteratorType child_end(const NodeType *N) { return N->end(); }
};

} // End llvm namespace

#endif
