//===- Interval.cpp - Interval class code ---------------------------------===//
//
// This file contains the definition of the Interval class, which represents a
// partition of a control flow graph of some kind.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Interval.h"
#include "llvm/BasicBlock.h"
#include "llvm/Support/CFG.h"
#include <algorithm>

//===----------------------------------------------------------------------===//
// Interval Implementation
//===----------------------------------------------------------------------===//

// isLoop - Find out if there is a back edge in this interval...
//
bool Interval::isLoop() const {
  // There is a loop in this interval iff one of the predecessors of the header
  // node lives in the interval.
  for (::pred_iterator I = ::pred_begin(HeaderNode), E = ::pred_end(HeaderNode);
       I != E; ++I) {
    if (contains(*I)) return true;
  }
  return false;
}


void Interval::print(std::ostream &o) const {
  o << "-------------------------------------------------------------\n"
       << "Interval Contents:\n";
  
  // Print out all of the basic blocks in the interval...
  std::copy(Nodes.begin(), Nodes.end(), 
            std::ostream_iterator<BasicBlock*>(o, "\n"));

  o << "Interval Predecessors:\n";
  std::copy(Predecessors.begin(), Predecessors.end(), 
            std::ostream_iterator<BasicBlock*>(o, "\n"));
  
  o << "Interval Successors:\n";
  std::copy(Successors.begin(), Successors.end(), 
            std::ostream_iterator<BasicBlock*>(o, "\n"));
}
