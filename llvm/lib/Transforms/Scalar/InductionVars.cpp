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
#include "llvm/Tools/STLExtras.h"

static bool ProcessInterval(cfg::Interval *Int) {
  if (!Int->isLoop()) return false;  // Not a loop?  Ignore it!

  cerr << "Found Looping Interval: " << Int; //->HeaderNode;
  return false;
}

static bool ProcessIntervalPartition(cfg::IntervalPartition &IP) {
  // This currently just prints out information about the interval structure
  // of the method...
  static unsigned N = 0;
  cerr << "\n***********Interval Partition #" << (++N) << "************\n\n";
  copy(IP.begin(), IP.end(), ostream_iterator<cfg::Interval*>(cerr, "\n"));

  cerr << "\n*********** PERFORMING WORK ************\n\n";

  // Loop over all of the intervals in the partition and look for induction
  // variables in intervals that represent loops.
  //
  return reduce_apply(IP.begin(), IP.end(), bitwise_or<bool>(), false,
		      ptr_fun(ProcessInterval));
}

// DoInductionVariableCannonicalize - Simplify induction variables in loops
//
bool DoInductionVariableCannonicalize(Method *M) {
  cfg::IntervalPartition *IP = new cfg::IntervalPartition(M);
  bool Changed = false;

  while (!IP->isDegeneratePartition()) {
    Changed |= ProcessIntervalPartition(*IP);

    // Calculate the reduced version of this graph until we get to an 
    // irreducible graph or a degenerate graph...
    //
    cfg::IntervalPartition *NewIP = new cfg::IntervalPartition(*IP, false);
    if (NewIP->size() == IP->size()) {
      cerr << "IRREDUCIBLE GRAPH FOUND!!!\n";
      return Changed;
    }
    delete IP;
    IP = NewIP;
  }

  delete IP;
  return Changed;
}
