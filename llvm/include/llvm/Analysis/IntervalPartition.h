//===- IntervalPartition.h - Interval partition Calculation ------*- C++ -*--=//
//
// This file contains the declaration of the cfg::IntervalPartition class, which
// calculates and represents the interval partition of a method, or a
// preexisting interval partition.
//
// In this way, the interval partition may be used to reduce a flow graph down
// to its degenerate single node interval partition (unless it is irreducible).
//
// TODO: The IntervalPartition class should take a bool parameter that tells
// whether it should add the "tails" of an interval to an interval itself or if
// they should be represented as distinct intervals.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INTERVAL_PARTITION_H
#define LLVM_INTERVAL_PARTITION_H

#include "llvm/Analysis/Interval.h"
#include <map>

class Method;

namespace cfg {

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
