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
class IntervalPartition : public vector<Interval*> {
  typedef map<BasicBlock*, Interval*> IntervalMapTy;
  IntervalMapTy IntervalMap;

  typedef vector<Interval*> IntervalListTy;
  Interval *RootInterval;

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

private:
  // addIntervalToPartition - Add an interval to the internal list of intervals,
  // and then add mappings from all of the basic blocks in the interval to the
  // interval itself (in the IntervalMap).
  //
  void addIntervalToPartition(Interval *I);

  // updatePredecessors - Interval generation only sets the successor fields of
  // the interval data structures.  After interval generation is complete,
  // run through all of the intervals and propogate successor info as
  // predecessor info.
  //
  void updatePredecessors(Interval *Int);
};

}    // End namespace cfg

#endif
