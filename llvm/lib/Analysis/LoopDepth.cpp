//===- LoopDepth.cpp - Loop Depth Calculation --------------------*- C++ -*--=//
//
// This file provides a simple class to calculate the loop depth of a 
// BasicBlock.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopDepth.h"
#include "llvm/Analysis/IntervalPartition.h"
#include "llvm/Support/STLExtras.h"
#include <algorithm>

inline void LoopDepthCalculator::AddBB(const BasicBlock *BB) {
  ++LoopDepth[BB];   // Increment the loop depth count for the specified BB
}

inline void LoopDepthCalculator::ProcessInterval(cfg::Interval *I) {
  if (!I->isLoop()) return;  // Ignore nonlooping intervals...

  for_each(I->Nodes.begin(), I->Nodes.end(), 
	   bind_obj(this, &LoopDepthCalculator::AddBB));
}

LoopDepthCalculator::LoopDepthCalculator(Method *M) {
  //map<const BasicBlock*, unsigned> LoopDepth;

  cfg::IntervalPartition *IP = new cfg::IntervalPartition(M);
  while (!IP->isDegeneratePartition()) {
    for_each(IP->begin(), IP->end(), 
	     bind_obj(this, &LoopDepthCalculator::ProcessInterval));

    // Calculate the reduced version of this graph until we get to an 
    // irreducible graph or a degenerate graph...
    //
    cfg::IntervalPartition *NewIP = new cfg::IntervalPartition(*IP, true);
    if (NewIP->size() == IP->size()) {
      cerr << "IRREDUCIBLE GRAPH FOUND!!!\n";
      // TODO: fix irreducible graph
      return;
    }
    delete IP;
    IP = NewIP;
  }

  delete IP;
}
