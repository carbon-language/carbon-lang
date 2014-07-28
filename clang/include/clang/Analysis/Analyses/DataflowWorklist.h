//===- DataflowWorklist.h - worklist for dataflow analysis --------*- C++ --*-//
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

#ifndef LLVM_CLANG_DATAFLOW_WORKLIST
#define LLVM_CLANG_DATAFLOW_WORKLIST

#include "clang/Analysis/Analyses/PostOrderCFGView.h"

namespace clang {

class DataflowWorklistBase {
protected:
  PostOrderCFGView::iterator PO_I, PO_E;
  PostOrderCFGView::BlockOrderCompare comparator;
  SmallVector<const CFGBlock *, 20> worklist;
  llvm::BitVector enqueuedBlocks;

  DataflowWorklistBase(const CFG &cfg, PostOrderCFGView &view)
    : PO_I(view.begin()), PO_E(view.end()),
      comparator(view.getComparator()),
      enqueuedBlocks(cfg.getNumBlockIDs(), true) {
        // Treat the first block as already analyzed.
        if (PO_I != PO_E) {
          assert(*PO_I == &cfg.getEntry());
          enqueuedBlocks[(*PO_I)->getBlockID()] = false;
          ++PO_I;
        }
      }
};

class DataflowWorklist : DataflowWorklistBase {

public:
  DataflowWorklist(const CFG &cfg, AnalysisDeclContext &Ctx)
    : DataflowWorklistBase(cfg, *Ctx.getAnalysis<PostOrderCFGView>()) {}

  void enqueueBlock(const CFGBlock *block);
  void enqueuePredecessors(const CFGBlock *block);
  void enqueueSuccessors(const CFGBlock *block);
  const CFGBlock *dequeue();

  void sortWorklist();
};

} // end clang namespace

#endif
