//===-- IntervalWriter.cpp - Library for printing Intervals ------*- C++ -*--=//
//
// This library implements the interval printing functionality defined in 
// llvm/Assembly/Writer.h
//
//===----------------------------------------------------------------------===//

#include "llvm/Assembly/Writer.h"
#include "llvm/Analysis/Interval.h"
#include <iterator>
#include <algorithm>

void cfg::WriteToOutput(const Interval *I, ostream &o) {
  o << "-------------------------------------------------------------\n"
       << "Interval Contents:\n";
  
  // Print out all of the basic blocks in the interval...
  copy(I->Nodes.begin(), I->Nodes.end(), 
       ostream_iterator<BasicBlock*>(o, "\n"));

  o << "Interval Predecessors:\n";
  copy(I->Predecessors.begin(), I->Predecessors.end(), 
       ostream_iterator<BasicBlock*>(o, "\n"));
  
  o << "Interval Successors:\n";
  copy(I->Successors.begin(), I->Successors.end(), 
       ostream_iterator<BasicBlock*>(o, "\n"));
}

#include "llvm/Analysis/Dominators.h"

ostream &operator<<(ostream &o, const set<const BasicBlock*> &BBs) {
  copy(BBs.begin(), BBs.end(), ostream_iterator<const BasicBlock*>(o, "\n"));
  return o;
}

void cfg::WriteToOutput(const DominatorSet &DS, ostream &o) {
  for (DominatorSet::const_iterator I = DS.begin(), E = DS.end(); I != E; ++I) {
    o << "=============================--------------------------------\n"
      << "\nDominator Set For Basic Block\n" << I->first
      << "-------------------------------\n" << I->second << endl;
  }
}


void cfg::WriteToOutput(const ImmediateDominators &ID, ostream &o) {
  for (ImmediateDominators::const_iterator I = ID.begin(), E = ID.end();
       I != E; ++I) {
    o << "=============================--------------------------------\n"
      << "\nImmediate Dominator For Basic Block\n" << I->first
      << "is: \n" << I->second << endl;
  }
}


void cfg::WriteToOutput(const DominatorTree &DT, ostream &o) {

}

void cfg::WriteToOutput(const DominanceFrontier &DF, ostream &o) {
  for (DominanceFrontier::const_iterator I = DF.begin(), E = DF.end();
       I != E; ++I) {
    o << "=============================--------------------------------\n"
      << "\nDominance Frontier For Basic Block\n" << I->first
      << "is: \n" << I->second << endl;
  }
}

