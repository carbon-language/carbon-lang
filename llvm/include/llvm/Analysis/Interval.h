//===- llvm/Analysis/Intervals.h - Interval partition Calculation-*- C++ -*--=//
//
// This file contains the declaration of the cfg::IntervalPartition class, which
// calculates and represents the interval partition of a method, or a
// preexisting interval partition.
//
// In this way, the interval partition may be used to reduce a flow graph down
// to its degenerate single node interval partition (unless it is irreducible).
//
// TODO: Provide an interval iterator that codifies the internals of 
// IntervalPartition.  Inside, it would have a stack of Interval*'s, and would
// walk the interval partition in depth first order.  IntervalPartition would
// then be a client of this iterator.  The iterator should work on Method*,
// const Method*, IntervalPartition*, and const IntervalPartition*.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INTERVALS_H
#define LLVM_INTERVALS_H

#include <vector>
#include <map>
#include <algorithm>

class Method;
class BasicBlock;

namespace cfg {

class IntervalPartition;

//===----------------------------------------------------------------------===//
//
// Interval Class - An Interval is a set of nodes defined such that every node
// in the interval has all of its predecessors in the interval (except for the
// header)
//
class Interval {
  friend class IntervalPartition;

  // HeaderNode - The header BasicBlock, which dominates all BasicBlocks in this
  // interval.  Also, any loops in this interval must go through the HeaderNode.
  //
  BasicBlock *HeaderNode;
public:
  typedef vector<BasicBlock*>::iterator succ_iterator;
  typedef vector<BasicBlock*>::iterator pred_iterator;
  typedef vector<BasicBlock*>::iterator node_iterator;

  inline BasicBlock *getHeaderNode() const { return HeaderNode; }

  // Nodes - The basic blocks in this interval.
  //
  vector<BasicBlock*> Nodes;

  // Successors - List of BasicBlocks that are reachable directly from nodes in
  // this interval, but are not in the interval themselves.
  // These nodes neccesarily must be header nodes for other intervals.
  //
  vector<BasicBlock*> Successors;

  // Predecessors - List of BasicBlocks that have this Interval's header block
  // as one of their successors.
  //
  vector<BasicBlock*> Predecessors;

  // contains - Find out if a basic block is in this interval
  inline bool contains(BasicBlock *BB) const {
    return find(Nodes.begin(), Nodes.end(), BB) != Nodes.end();
  }

  // isSuccessor - find out if a basic block is a successor of this Interval
  inline bool isSuccessor(BasicBlock *BB) const {
    return find(Successors.begin(), Successors.end(), BB) != Successors.end();
  }

  // isLoop - Find out if there is a back edge in this interval...
  bool isLoop() const;

private:           // Only accessable by IntervalPartition class
  inline Interval(BasicBlock *Header) : HeaderNode(Header) {
    Nodes.push_back(Header);
  }
};


// succ_begin/succ_end - define global functions so that Intervals may be used
// just like BasicBlocks can with the succ_* functions, and *::succ_iterator.
//
inline Interval::succ_iterator succ_begin(Interval *I) { 
  return I->Successors.begin();
}
inline Interval::succ_iterator succ_end(Interval *I) { 
  return I->Successors.end();
}

// pred_begin/pred_end - define global functions so that Intervals may be used
// just like BasicBlocks can with the pred_* functions, and *::pred_iterator.
//
inline Interval::pred_iterator pred_begin(Interval *I) { 
  return I->Predecessors.begin();
}
inline Interval::pred_iterator pred_end(Interval *I) { 
  return I->Predecessors.end();
}


//===----------------------------------------------------------------------===//
//
// IntervalPartition - This class builds and holds an "interval partition" for
// a method.  This partition divides the control flow graph into a set of
// maximal intervals, as defined with the properties above.  Intuitively, a
// BasicBlock is a (possibly nonexistent) loop with a "tail" of non looping
// nodes following it.
//
class IntervalPartition {
  typedef map<BasicBlock*, Interval*> IntervalMapTy;
  IntervalMapTy IntervalMap;

  typedef vector<Interval*> IntervalListTy;
  IntervalListTy IntervalList;
  Interval *RootInterval;

public:
  typedef IntervalListTy::iterator iterator;

public:
  // IntervalPartition ctor - Build the partition for the specified method
  IntervalPartition(Method *M);

  // IntervalPartition ctor - Build a reduced interval partition from an
  // existing interval graph.  This takes an additional boolean parameter to
  // distinguish it from a copy constructor.  Always pass in false for now.
  //
  IntervalPartition(IntervalPartition &I, bool);

  // Destructor - Free memory
  ~IntervalPartition();

  // getRootInterval() - Return the root interval that contains the starting
  // block of the method.
  inline Interval *getRootInterval() { return RootInterval; }

  // isDegeneratePartition() - Returns true if the interval partition contains
  // a single interval, and thus cannot be simplified anymore.
  bool isDegeneratePartition() { return size() == 1; }

  // TODO: isIrreducible - look for triangle graph.

  // getBlockInterval - Return the interval that a basic block exists in.
  inline Interval *getBlockInterval(BasicBlock *BB) {
    IntervalMapTy::iterator I = IntervalMap.find(BB);
    return I != IntervalMap.end() ? I->second : 0;
  }

  // Iterators to iterate over all of the intervals in the method
  inline iterator begin() { return IntervalList.begin(); }
  inline iterator end()   { return IntervalList.end(); }
  inline unsigned size()  { return IntervalList.size(); }

private:
  // ProcessInterval - This method is used during the construction of the 
  // interval graph.  It walks through the source graph, recursively creating
  // an interval per invokation until the entire graph is covered.  This uses
  // the ProcessNode method to add all of the nodes to the interval.
  //
  // This method is templated because it may operate on two different source
  // graphs: a basic block graph, or a preexisting interval graph.
  //
  template<class NodeTy, class OrigContainer>
  void ProcessInterval(NodeTy *Node, OrigContainer *OC);

  // ProcessNode - This method is called by ProcessInterval to add nodes to the
  // interval being constructed, and it is also called recursively as it walks
  // the source graph.  A node is added to the current interval only if all of
  // its predecessors are already in the graph.  This also takes care of keeping
  // the successor set of an interval up to date.
  //
  // This method is templated because it may operate on two different source
  // graphs: a basic block graph, or a preexisting interval graph.
  //
  template<class NodeTy, class OrigContainer>
  void ProcessNode(Interval *Int, NodeTy *Node, OrigContainer *OC);

  // addNodeToInterval - This method exists to assist the generic ProcessNode
  // with the task of adding a node to the new interval, depending on the 
  // type of the source node.  In the case of a CFG source graph (BasicBlock 
  // case), the BasicBlock itself is added to the interval.  In the case of
  // an IntervalPartition source graph (Interval case), all of the member
  // BasicBlocks are added to the interval.
  //
  inline void addNodeToInterval(Interval *Int, Interval *I);
  inline void addNodeToInterval(Interval *Int, BasicBlock *BB);

  // updatePredecessors - Interval generation only sets the successor fields of
  // the interval data structures.  After interval generation is complete,
  // run through all of the intervals and propogate successor info as
  // predecessor info.
  //
  void updatePredecessors(Interval *Int);
};

}    // End namespace cfg

#endif
