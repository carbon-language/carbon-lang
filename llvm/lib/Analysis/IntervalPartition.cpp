//===- IntervalPartition.cpp - Interval Partition module code ----*- C++ -*--=//
//
// This file contains the definition of the cfg::IntervalPartition class, which
// calculates and represent the interval partition of a method.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/IntervalIterator.h"

using namespace cfg;

//===----------------------------------------------------------------------===//
// IntervalPartition Implementation
//===----------------------------------------------------------------------===//

template <class T> static inline void deleter(T *Ptr) { delete Ptr; }

// Destructor - Free memory
IntervalPartition::~IntervalPartition() {
  for_each(begin(), end(), deleter<cfg::Interval>);
}

// addIntervalToPartition - Add an interval to the internal list of intervals,
// and then add mappings from all of the basic blocks in the interval to the
// interval itself (in the IntervalMap).
//
void IntervalPartition::addIntervalToPartition(Interval *I) {
  IntervalList.push_back(I);

  // Add mappings for all of the basic blocks in I to the IntervalPartition
  for (Interval::node_iterator It = I->Nodes.begin(), End = I->Nodes.end();
       It != End; ++It)
    IntervalMap.insert(make_pair(*It, I));
}

// updatePredecessors - Interval generation only sets the successor fields of
// the interval data structures.  After interval generation is complete,
// run through all of the intervals and propogate successor info as
// predecessor info.
//
void IntervalPartition::updatePredecessors(cfg::Interval *Int) {
  BasicBlock *Header = Int->getHeaderNode();
  for (Interval::succ_iterator I = Int->Successors.begin(), 
	                       E = Int->Successors.end(); I != E; ++I)
    getBlockInterval(*I)->Predecessors.push_back(Header);
}

// IntervalPartition ctor - Build the first level interval partition for the
// specified method...
//
IntervalPartition::IntervalPartition(Method *M) {
  assert(M->getBasicBlocks().front() && "Cannot operate on prototypes!");

  // Pass false to intervals_begin because we take ownership of it's memory
  method_interval_iterator I = intervals_begin(M, false);
  method_interval_iterator End = intervals_end(M);
  assert(I != End && "No intervals in method!?!?!");

  addIntervalToPartition(RootInterval = *I);

  for (++I; I != End; ++I)
    addIntervalToPartition(*I);

  // Now that we know all of the successor information, propogate this to the
  // predecessors for each block...
  for(iterator It = begin(), E = end(); It != E; ++It)
    updatePredecessors(*It);
}


// IntervalPartition ctor - Build a reduced interval partition from an
// existing interval graph.  This takes an additional boolean parameter to
// distinguish it from a copy constructor.  Always pass in false for now.
//
IntervalPartition::IntervalPartition(IntervalPartition &IP, bool) {
  Interval *MethodStart = IP.getRootInterval();
  assert(MethodStart && "Cannot operate on empty IntervalPartitions!");

  // Pass false to intervals_begin because we take ownership of it's memory
  interval_part_interval_iterator I = intervals_begin(IP, false);
  interval_part_interval_iterator End = intervals_end(IP);
  assert(I != End && "No intervals in interval partition!?!?!");

  addIntervalToPartition(RootInterval = *I);

  for (++I; I != End; ++I)
    addIntervalToPartition(*I);

  // Now that we know all of the successor information, propogate this to the
  // predecessors for each block...
  for(iterator I = begin(), E = end(); I != E; ++I)
    updatePredecessors(*I);
}
