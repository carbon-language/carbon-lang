//===- Interval.cpp - Interval class code ------------------------*- C++ -*--=//
//
// This file contains the definition of the cfg::Interval class, which
// represents a partition of a control flow graph of some kind.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Interval.h"
#include "llvm/BasicBlock.h"

//===----------------------------------------------------------------------===//
// Interval Implementation
//===----------------------------------------------------------------------===//

// isLoop - Find out if there is a back edge in this interval...
//
bool cfg::Interval::isLoop() const {
  // There is a loop in this interval iff one of the predecessors of the header
  // node lives in the interval.
  for (BasicBlock::pred_iterator I = HeaderNode->pred_begin(), 
                                 E = HeaderNode->pred_end(); I != E; ++I) {
    if (contains(*I)) return true;
  }
  return false;
}


