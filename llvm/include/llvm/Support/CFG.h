//===-- llvm/Support/CFG.h - Process LLVM structures as graphs ---*- C++ -*--=//
//
// This file defines specializations of GraphTraits that allow Methods and
// BasicBlock graphs to be treated as proper graphs for generic algorithms.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CFG_H
#define LLVM_CFG_H

#include "Support/GraphTraits.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"

//===--------------------------------------------------------------------===//
// GraphTraits specializations for basic block graphs (CFGs)
//===--------------------------------------------------------------------===//

// Provide specializations of GraphTraits to be able to treat a method as a 
// graph of basic blocks...

template <> struct GraphTraits<BasicBlock*> {
  typedef BasicBlock NodeType;
  typedef BasicBlock::succ_iterator ChildIteratorType;

  static NodeType *getEntryNode(BasicBlock *BB) { return BB; }
  static inline ChildIteratorType child_begin(NodeType *N) { 
    return N->succ_begin(); 
  }
  static inline ChildIteratorType child_end(NodeType *N) { 
    return N->succ_end(); 
  }
};

template <> struct GraphTraits<const BasicBlock*> {
  typedef const BasicBlock NodeType;
  typedef BasicBlock::succ_const_iterator ChildIteratorType;

  static NodeType *getEntryNode(const BasicBlock *BB) { return BB; }

  static inline ChildIteratorType child_begin(NodeType *N) { 
    return N->succ_begin(); 
  }
  static inline ChildIteratorType child_end(NodeType *N) { 
    return N->succ_end(); 
  }
};

// Provide specializations of GraphTraits to be able to treat a method as a 
// graph of basic blocks... and to walk it in inverse order.  Inverse order for
// a method is considered to be when traversing the predecessor edges of a BB
// instead of the successor edges.
//
template <> struct GraphTraits<Inverse<BasicBlock*> > {
  typedef BasicBlock NodeType;
  typedef BasicBlock::pred_iterator ChildIteratorType;
  static NodeType *getEntryNode(Inverse<BasicBlock *> G) { return G.Graph; }
  static inline ChildIteratorType child_begin(NodeType *N) { 
    return N->pred_begin(); 
  }
  static inline ChildIteratorType child_end(NodeType *N) { 
    return N->pred_end(); 
  }
};

template <> struct GraphTraits<Inverse<const BasicBlock*> > {
  typedef const BasicBlock NodeType;
  typedef BasicBlock::pred_const_iterator ChildIteratorType;
  static NodeType *getEntryNode(Inverse<const BasicBlock*> G) {
    return G.Graph; 
  }
  static inline ChildIteratorType child_begin(NodeType *N) { 
    return N->pred_begin(); 
  }
  static inline ChildIteratorType child_end(NodeType *N) { 
    return N->pred_end(); 
  }
};



//===--------------------------------------------------------------------===//
// GraphTraits specializations for method basic block graphs (CFGs)
//===--------------------------------------------------------------------===//

// Provide specializations of GraphTraits to be able to treat a method as a 
// graph of basic blocks... these are the same as the basic block iterators,
// except that the root node is implicitly the first node of the method.
//
template <> struct GraphTraits<Method*> : public GraphTraits<BasicBlock*> {
  static NodeType *getEntryNode(Method *M) { return M->front(); }
};
template <> struct GraphTraits<const Method*> :
  public GraphTraits<const BasicBlock*> {
  static NodeType *getEntryNode(const Method *M) { return M->front(); }
};


// Provide specializations of GraphTraits to be able to treat a method as a 
// graph of basic blocks... and to walk it in inverse order.  Inverse order for
// a method is considered to be when traversing the predecessor edges of a BB
// instead of the successor edges.
//
template <> struct GraphTraits<Inverse<Method*> > :
  public GraphTraits<Inverse<BasicBlock*> > {
  static NodeType *getEntryNode(Inverse<Method *> G) { return G.Graph->front();}
};
template <> struct GraphTraits<Inverse<const Method*> > :
  public GraphTraits<Inverse<const BasicBlock*> > {
  static NodeType *getEntryNode(Inverse<const Method *> G) {
    return G.Graph->front();
  }
};

#endif
