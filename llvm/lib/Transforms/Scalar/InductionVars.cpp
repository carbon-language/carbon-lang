//===- InductionVars.cpp - Induction Variable Cannonicalization code --------=//
//
// This file implements induction variable cannonicalization of loops.
//
// Specifically, after this executes, the following is true:
//   - There is a single induction variable for each loop (that used to contain
//     at least one induction variable)
//   - This induction variable starts at 0 and steps by 1 per iteration
//   - All other preexisting induction variables are adjusted to operate in
//     terms of this primary induction variable
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Intervals.h"
#include "llvm/Opt/AllOpts.h"
#include "llvm/Assembly/Writer.h"

static void PrintIntervalInfo(cfg::Interval *I) {
  cerr << "-------------------------------------------------------------\n"
       << "Interval Contents:\n";
  
  // Print out all of the basic blocks in the interval...
  copy(I->Nodes.begin(), I->Nodes.end(), 
       ostream_iterator<BasicBlock*>(cerr, "\n"));

  cerr << "Interval Predecessors:\n";
  copy(I->Predecessors.begin(), I->Predecessors.end(), 
       ostream_iterator<BasicBlock*>(cerr, "\n"));
  
  cerr << "Interval Successors:\n";
  copy(I->Successors.begin(), I->Successors.end(), 
       ostream_iterator<BasicBlock*>(cerr, "\n"));
}

// DoInductionVariableCannonicalize - Simplify induction variables in loops
//
bool DoInductionVariableCannonicalize(Method *M) {
  cfg::IntervalPartition Intervals(M);

  // This currently just prints out information about the interval structure
  // of the method...
  for_each(Intervals.begin(), Intervals.end(), PrintIntervalInfo);

  cerr << "*************Reduced Interval**************\n\n";

  cfg::IntervalPartition Intervals2(Intervals, false);
  
  // This currently just prints out information about the interval structure
  // of the method...
  for_each(Intervals2.begin(), Intervals2.end(), PrintIntervalInfo);

  return false;
}
