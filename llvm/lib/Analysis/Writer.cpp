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
