//===- Interval.cpp - Interval class code ------------------------*- C++ -*--=//
//
// This file contains the definition of the cfg::Interval class, which
// represents a partition of a control flow graph of some kind.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Interval.h"
#include "llvm/BasicBlock.h"
#include "llvm/Support/CFG.h"

//===----------------------------------------------------------------------===//
// Interval Implementation
//===----------------------------------------------------------------------===//

// isLoop - Find out if there is a back edge in this interval...
//
bool cfg::Interval::isLoop() const {
  // There is a loop in this interval iff one of the predecessors of the header
  // node lives in the interval.
  for (::pred_iterator I = ::pred_begin(HeaderNode), E = ::pred_end(HeaderNode);
       I != E; ++I) {
    if (contains(*I)) return true;
  }
  return false;
}


