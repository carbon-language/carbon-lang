//===- DataflowWorklist.h - worklist for dataflow analysis --------*- C++ --*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// DataflowWorklist keeps track of blocks for dataflow analysis. It maintains a
// vector of blocks for priority processing, and falls back upon a reverse
// post-order iterator. It supports both forward (used in UninitializedValues)
// and backward (used in LiveVariables) analyses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_DATAFLOWWORKLIST_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_DATAFLOWWORKLIST_H

#include "clang/Analysis/Analyses/PostOrderCFGView.h"

namespace clang {

class DataflowWorklist {
  PostOrderCFGView::iterator PO_I, PO_E;
  SmallVector<const CFGBlock *, 20> worklist;
  llvm::BitVector enqueuedBlocks;

protected:
  DataflowWorklist(const CFG &cfg, PostOrderCFGView &view)
    : PO_I(view.begin()), PO_E(view.end()),
      enqueuedBlocks(cfg.getNumBlockIDs(), true) {
        // For forward analysis, treat the first block as already analyzed.
        if ((PO_I != PO_E) && (*PO_I == &cfg.getEntry())) {
          enqueuedBlocks[(*PO_I)->getBlockID()] = false;
          ++PO_I;
        }
      }

public:
  void enqueueBlock(const CFGBlock *block);
  void enqueuePredecessors(const CFGBlock *block);
  void enqueueSuccessors(const CFGBlock *block);
  const CFGBlock *dequeue();
};

class BackwardDataflowWorklist : public DataflowWorklist {
public:
  BackwardDataflowWorklist(const CFG &cfg, AnalysisDeclContext &Ctx)
    : DataflowWorklist(cfg, *Ctx.getAnalysis<PostOrderCFGView>()) {}
};

class ForwardDataflowWorklist : public DataflowWorklist {
public:
  ForwardDataflowWorklist(const CFG &cfg, AnalysisDeclContext &Ctx)
    : DataflowWorklist(cfg, *Ctx.getAnalysis<ReversePostOrderCFGView>()) {}
};

} // end clang namespace

#endif
