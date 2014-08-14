//===- DataflowWorklist.cpp - worklist for dataflow analysis ------*- C++ --*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// DataflowWorklist is used in LiveVariables and UninitializedValues analyses
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/DataflowWorklist.h"

using namespace clang;

// Marking a block as enqueued means that it cannot be re-added to the worklist,
// but it doesn't control when the algorithm terminates.
// Initially, enqueuedBlocks is set to true for all blocks;
// that's not because everything is added initially to the worklist,
// but instead, to cause the analysis to follow the initial graph traversal
// until we enqueue something on the worklist. 
void DataflowWorklist::enqueueBlock(const clang::CFGBlock *block) {
  if (block && !enqueuedBlocks[block->getBlockID()]) {
    enqueuedBlocks[block->getBlockID()] = true;
    worklist.push_back(block);
  }
}

// The analysis alternates between essentially two worklists.
// A prioritization worklist (SmallVector<const CFGBlock *> worklist)
// is consulted first, and if it's empty, we consult
// PostOrderCFGView::iterator PO_I, which implements either post-order traversal
// for backward analysis, or reverse post-order traversal for forward analysis.
// The prioritization worklist is used to prioritize analyzing from
// the beginning, or to prioritize updates fed by back edges.
// Typically, what gets enqueued on the worklist are back edges, which
// we want to prioritize analyzing first, because that causes dataflow facts
// to flow up the graph, which we then want to propagate forward.
// In practice this can cause the analysis to converge much faster.  
void DataflowWorklist::enqueueSuccessors(const clang::CFGBlock *block) {
  for (CFGBlock::const_succ_iterator I = block->succ_begin(),
       E = block->succ_end(); I != E; ++I) {
    enqueueBlock(*I);
  }
}

void DataflowWorklist::enqueuePredecessors(const clang::CFGBlock *block) {
  for (CFGBlock::const_pred_iterator I = block->pred_begin(),
       E = block->pred_end(); I != E; ++I) {
    enqueueBlock(*I);
  }
}

const CFGBlock *DataflowWorklist::dequeue() {
  const CFGBlock *B = nullptr;

  // First dequeue from the worklist.  This can represent
  // updates along backedges that we want propagated as quickly as possible.
  if (!worklist.empty())
    B = worklist.pop_back_val();

  // Next dequeue from the initial graph traversal (either post order or
  // reverse post order).  This is the theoretical ideal in the presence
  // of no back edges.
  else if (PO_I != PO_E) {
    B = *PO_I;
    ++PO_I;
  }
  else {
    return nullptr;
  }

  assert(enqueuedBlocks[B->getBlockID()] == true);
  enqueuedBlocks[B->getBlockID()] = false;
  return B;
}

