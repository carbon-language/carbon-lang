//===- llvm/Analysis/Intervals.h - Interval partition Calculation-*- C++ -*--=//
//
// This file contains the declaration of the cfg::IntervalPartition class, which
// calculates and represent the interval partition of a method.
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

// Interval Class - An Interval is a set of nodes defined such that every node
// in the interval has all of its predecessors in the interval (except for the
// header)
class Interval {
  friend class IntervalPartition;
public:
  typedef vector<BasicBlock*>::iterator succ_iterator;
  typedef vector<BasicBlock*>::iterator pred_iterator;
  typedef vector<BasicBlock*>::iterator node_iterator;

  // HeaderNode - The header BasicBlock, which dominates all BasicBlocks in this
  // interval.  Also, any loops in this interval must go through the HeaderNode.
  //
  BasicBlock *HeaderNode;

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

  inline bool contains(BasicBlock *BB) {
    return find(Nodes.begin(), Nodes.end(), BB) != Nodes.end();
  }

  inline bool isSuccessor(BasicBlock *BB) {
    return find(Successors.begin(), Successors.end(), BB) != Successors.end();
  }

private:           // Only accessable by IntervalPartition class
  inline Interval(BasicBlock *Header) : HeaderNode(Header) {
    Nodes.push_back(Header);
  }
};


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

  // getRootInterval() - Return the root interval that contains the starting
  // block of the method
  inline Interval *getRootInterval() { return RootInterval; }

  inline Interval *getBlockInterval(BasicBlock *BB) {
    IntervalMapTy::iterator I = IntervalMap.find(BB);
    if (I != IntervalMap.end()) 
      return I->second;
    else
      return 0;
  }

  // Iterators to iterate over all of the intervals in the method
  inline iterator begin() { return IntervalList.begin(); }
  inline iterator end()   { return IntervalList.end(); }

private:
  void ProcessInterval(BasicBlock *Header);
  void ProcessBasicBlock(Interval *I, BasicBlock *BB);
  void UpdateSuccessors(Interval *Int);
};

}    // End namespace cfg

#endif
