//===- IntervalPartition.cpp - Interval Partition module code -------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the definition of the IntervalPartition class, which
// calculates and represent the interval partition of a function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/IntervalIterator.h"
#include "Support/STLExtras.h"

namespace llvm {

static RegisterAnalysis<IntervalPartition>
X("intervals", "Interval Partition Construction", true);

//===----------------------------------------------------------------------===//
// IntervalPartition Implementation
//===----------------------------------------------------------------------===//

// destroy - Reset state back to before function was analyzed
void IntervalPartition::destroy() {
  for_each(Intervals.begin(), Intervals.end(), deleter<Interval>);
  IntervalMap.clear();
  RootInterval = 0;
}

void IntervalPartition::print(std::ostream &O) const {
  std::copy(Intervals.begin(), Intervals.end(),
            std::ostream_iterator<const Interval *>(O, "\n"));
}

// addIntervalToPartition - Add an interval to the internal list of intervals,
// and then add mappings from all of the basic blocks in the interval to the
// interval itself (in the IntervalMap).
//
void IntervalPartition::addIntervalToPartition(Interval *I) {
  Intervals.push_back(I);

  // Add mappings for all of the basic blocks in I to the IntervalPartition
  for (Interval::node_iterator It = I->Nodes.begin(), End = I->Nodes.end();
       It != End; ++It)
    IntervalMap.insert(std::make_pair(*It, I));
}

// updatePredecessors - Interval generation only sets the successor fields of
// the interval data structures.  After interval generation is complete,
// run through all of the intervals and propagate successor info as
// predecessor info.
//
void IntervalPartition::updatePredecessors(Interval *Int) {
  BasicBlock *Header = Int->getHeaderNode();
  for (Interval::succ_iterator I = Int->Successors.begin(), 
	                       E = Int->Successors.end(); I != E; ++I)
    getBlockInterval(*I)->Predecessors.push_back(Header);
}

// IntervalPartition ctor - Build the first level interval partition for the
// specified function...
//
bool IntervalPartition::runOnFunction(Function &F) {
  // Pass false to intervals_begin because we take ownership of it's memory
  function_interval_iterator I = intervals_begin(&F, false);
  assert(I != intervals_end(&F) && "No intervals in function!?!?!");

  addIntervalToPartition(RootInterval = *I);

  ++I;  // After the first one...

  // Add the rest of the intervals to the partition...
  for_each(I, intervals_end(&F),
	   bind_obj(this, &IntervalPartition::addIntervalToPartition));

  // Now that we know all of the successor information, propagate this to the
  // predecessors for each block...
  for_each(Intervals.begin(), Intervals.end(), 
	   bind_obj(this, &IntervalPartition::updatePredecessors));
  return false;
}


// IntervalPartition ctor - Build a reduced interval partition from an
// existing interval graph.  This takes an additional boolean parameter to
// distinguish it from a copy constructor.  Always pass in false for now.
//
IntervalPartition::IntervalPartition(IntervalPartition &IP, bool) {
  Interval *FunctionStart = IP.getRootInterval();
  assert(FunctionStart && "Cannot operate on empty IntervalPartitions!");

  // Pass false to intervals_begin because we take ownership of it's memory
  interval_part_interval_iterator I = intervals_begin(IP, false);
  assert(I != intervals_end(IP) && "No intervals in interval partition!?!?!");

  addIntervalToPartition(RootInterval = *I);

  ++I;  // After the first one...

  // Add the rest of the intervals to the partition...
  for_each(I, intervals_end(IP),
	   bind_obj(this, &IntervalPartition::addIntervalToPartition));

  // Now that we know all of the successor information, propagate this to the
  // predecessors for each block...
  for_each(Intervals.begin(), Intervals.end(), 
	   bind_obj(this, &IntervalPartition::updatePredecessors));
}

} // End llvm namespace
