//===- IntervalPartition.cpp - Interval Partition module code ----*- C++ -*--=//
//
// This file contains the definition of the cfg::IntervalPartition class, which
// calculates and represent the interval partition of a method.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/IntervalIterator.h"
#include "Support/STLExtras.h"

using namespace cfg;
using std::make_pair;

AnalysisID IntervalPartition::ID(AnalysisID::create<IntervalPartition>());

//===----------------------------------------------------------------------===//
// IntervalPartition Implementation
//===----------------------------------------------------------------------===//

// destroy - Reset state back to before method was analyzed
void IntervalPartition::destroy() {
  for_each(begin(), end(), deleter<cfg::Interval>);
  IntervalMap.clear();
  RootInterval = 0;
}

// addIntervalToPartition - Add an interval to the internal list of intervals,
// and then add mappings from all of the basic blocks in the interval to the
// interval itself (in the IntervalMap).
//
void IntervalPartition::addIntervalToPartition(Interval *I) {
  push_back(I);

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
bool IntervalPartition::runOnMethod(Method *M) {
  assert(M->front() && "Cannot operate on prototypes!");

  // Pass false to intervals_begin because we take ownership of it's memory
  method_interval_iterator I = intervals_begin(M, false);
  assert(I != intervals_end(M) && "No intervals in method!?!?!");

  addIntervalToPartition(RootInterval = *I);

  ++I;  // After the first one...

  // Add the rest of the intervals to the partition...
  for_each(I, intervals_end(M),
	   bind_obj(this, &IntervalPartition::addIntervalToPartition));

  // Now that we know all of the successor information, propogate this to the
  // predecessors for each block...
  for_each(begin(), end(), 
	   bind_obj(this, &IntervalPartition::updatePredecessors));
  return false;
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
  assert(I != intervals_end(IP) && "No intervals in interval partition!?!?!");

  addIntervalToPartition(RootInterval = *I);

  ++I;  // After the first one...

  // Add the rest of the intervals to the partition...
  for_each(I, intervals_end(IP),
	   bind_obj(this, &IntervalPartition::addIntervalToPartition));

  // Now that we know all of the successor information, propogate this to the
  // predecessors for each block...
  for_each(begin(), end(), 
	   bind_obj(this, &IntervalPartition::updatePredecessors));
}
