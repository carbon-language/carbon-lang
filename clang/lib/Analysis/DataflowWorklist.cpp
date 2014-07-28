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
// but instead, to cause the forward analysis to follow the reverse post order
// until we enqueue something on the worklist. 
void DataflowWorklist::enqueueBlock(const clang::CFGBlock *block) {
  if (block && !enqueuedBlocks[block->getBlockID()]) {
    enqueuedBlocks[block->getBlockID()] = true;
    worklist.push_back(block);
  }
}

// The forward analysis alternates between essentially two worklists.
// A prioritization worklist (SmallVector<const CFGBlock *> worklist)
// is consulted first, and if it's empty, we consult the reverse
// post-order traversal (PostOrderCFGView::iterator PO_I).
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

// The reverse analysis uses a simple re-sorting of the worklist to
// reprioritize it.  It's not as efficient as the two-worklists approach,
// but it isn't performance sensitive since it's used by the static analyzer,
// and the static analyzer does far more work that dwarfs the work done here.
// TODO: It would still be nice to use the same approach for both analyses.
void DataflowWorklist::enqueuePredecessors(const clang::CFGBlock *block) {
  const unsigned OldWorklistSize = worklist.size();
  for (CFGBlock::const_pred_iterator I = block->pred_begin(),
       E = block->pred_end(); I != E; ++I) {
    enqueueBlock(*I);
  }

  if (OldWorklistSize == 0 || OldWorklistSize == worklist.size())
    return;

  sortWorklist();
}

const CFGBlock *DataflowWorklist::dequeue() {
  const CFGBlock *B = nullptr;

  // First dequeue from the worklist.  This can represent
  // updates along backedges that we want propagated as quickly as possible.
  if (!worklist.empty())
    B = worklist.pop_back_val();

  // Next dequeue from the initial reverse post order.  This is the
  // theoretical ideal in the presence of no back edges.
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

void DataflowWorklist::sortWorklist() {
  std::sort(worklist.begin(), worklist.end(), comparator);
}

