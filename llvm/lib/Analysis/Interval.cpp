//===- Intervals.cpp - Interval partition Calculation ------------*- C++ -*--=//
//
// This file contains the declaration of the cfg::IntervalPartition class, which
// calculates and represent the interval partition of a method.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Intervals.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"
#include "llvm/CFG.h"

void cfg::IntervalPartition::UpdateSuccessors(cfg::Interval *Int) {
  BasicBlock *Header = Int->HeaderNode;
  for (cfg::Interval::succ_iterator I = Int->Successors.begin(), 
	                            E = Int->Successors.end(); I != E; ++I)
    getBlockInterval(*I)->Predecessors.push_back(Header);
}

// IntervalPartition ctor - Build the partition for the specified method
cfg::IntervalPartition::IntervalPartition(Method *M) {
  BasicBlock *MethodStart = M->getBasicBlocks().front();
  assert(MethodStart && "Cannot operate on prototypes!");

  ProcessInterval(MethodStart);
  RootInterval = getBlockInterval(MethodStart);

  // Now that we know all of the successor information, propogate this to the
  // predecessors for each block...
  for(iterator I = begin(), E = end(); I != E; ++I)
    UpdateSuccessors(*I);
}

void cfg::IntervalPartition::ProcessInterval(BasicBlock *Header) {
  if (getBlockInterval(Header)) return;  // Interval already constructed

  Interval *Int = new Interval(Header);
  IntervalList.push_back(Int);           // Add the interval to our current set
  IntervalMap.insert(make_pair(Header, Int));

  // Check all of our successors to see if they are in the interval...
  for (succ_iterator I = succ_begin(Header), E = succ_end(Header); I != E; ++I)
    ProcessBasicBlock(Int, *I);

  // Build all of the successor intervals of this interval now...
  for(Interval::succ_iterator I = Int->Successors.begin(), 
	E = Int->Successors.end(); I != E; ++I)
    ProcessInterval(*I);
}

void cfg::IntervalPartition::ProcessBasicBlock(Interval *Int, BasicBlock *BB) {
  assert(Int && "Null interval == bad!");
  assert(BB && "Null interval == bad!");

  Interval *CurInt = getBlockInterval(BB);
  if (CurInt == Int) {                  // Already in this interval...
    return;
  } else if (CurInt != 0) {             // In another interval, add as successor
    if (!Int->isSuccessor(BB))          // Add only if not already in set
      Int->Successors.push_back(BB);
  } else {                              // Otherwise, not in interval yet
    for (pred_iterator I = pred_begin(BB), E = pred_end(BB); I != E; ++I) {
      if (!Int->contains(*I)) {         // If pred not in interval, we can't be
	if (!Int->isSuccessor(BB))      // Add only if not already in set
	  Int->Successors.push_back(BB);
	return;                         // See you later
      }
    }
    
    // If we get here, then all of the predecessors of BB are in the interval
    // already.  In this case, we must add BB to the interval!
    Int->Nodes.push_back(BB);
    IntervalMap.insert(make_pair(BB, Int));
    
    if (Int->isSuccessor(BB)) {
      // If we were in the successor list from before... remove from succ list
      remove(Int->Successors.begin(), Int->Successors.end(), BB);
    }
    
    // Now that we have discovered that BB is in the interval, perhaps some of
    // its successors are as well?
    for (succ_iterator I = succ_begin(BB), E = succ_end(BB); I != E; ++I)
      ProcessBasicBlock(Int, *I);
  }
}
