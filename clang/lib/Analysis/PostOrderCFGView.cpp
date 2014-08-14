//===- PostOrderCFGView.cpp - Post order view of CFG blocks -------*- C++ --*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements post order views of the blocks in a CFG.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/PostOrderCFGView.h"

using namespace clang;

void PostOrderCFGView::anchor() { }

ReversePostOrderCFGView::ReversePostOrderCFGView(const CFG *cfg) {
  Blocks.reserve(cfg->getNumBlockIDs());
  CFGBlockSet BSet(cfg);

  typedef llvm::po_iterator<const CFG*, CFGBlockSet, true> po_iterator;

  for (po_iterator I = po_iterator::begin(cfg, BSet),
                   E = po_iterator::end(cfg, BSet); I != E; ++I) {
    Blocks.push_back(*I);      
    BlockOrder[*I] = Blocks.size();
  }
}

PostOrderCFGView *PostOrderCFGView::create(AnalysisDeclContext &ctx) {
  const CFG *cfg = ctx.getCFG();
  if (!cfg)
    return nullptr;
  return new PostOrderCFGView(cfg);
}

const void *PostOrderCFGView::getTag() { static int x; return &x; }

bool PostOrderCFGView::BlockOrderCompare::operator()(const CFGBlock *b1,
                                                     const CFGBlock *b2) const {
  PostOrderCFGView::BlockOrderTy::const_iterator b1It = POV.BlockOrder.find(b1);
  PostOrderCFGView::BlockOrderTy::const_iterator b2It = POV.BlockOrder.find(b2);
    
  unsigned b1V = (b1It == POV.BlockOrder.end()) ? 0 : b1It->second;
  unsigned b2V = (b2It == POV.BlockOrder.end()) ? 0 : b2It->second;
  return b1V > b2V;
}

PostOrderCFGView::PostOrderCFGView(const CFG *cfg) {
  unsigned size = cfg->getNumBlockIDs();
  Blocks.reserve(size);
  CFGBlockSet BSet(cfg);

  typedef llvm::po_iterator<const CFG*, CFGBlockSet, true,
                            llvm::GraphTraits<llvm::Inverse<const CFG*> >
                           > po_iterator;

  for (po_iterator I = po_iterator::begin(cfg, BSet),
                   E = po_iterator::end(cfg, BSet); I != E; ++I) {
    Blocks.push_back(*I);      
    BlockOrder[*I] = Blocks.size();
  }

  // It may be that some blocks are inaccessible going from the CFG exit upwards
  // (e.g. infinite loops); we still need to add them.
  for (CFG::const_iterator I = cfg->begin(), E = cfg->end();
       (Blocks.size() < size) && (I != E); ++I) {
    const CFGBlock* block = *I;
    // Add a chain going upwards.
    while (!BlockOrder.count(block)) {
      Blocks.push_back(block);
      BlockOrder[block] = Blocks.size();
      CFGBlock::const_pred_iterator PI = block->pred_begin(),
                                    PE = block->pred_end();
      for (; PI != PE; ++PI) {
        const CFGBlock* pred = *PI;
        if (pred && !BlockOrder.count(pred)) {
          block = pred;
          break;
        }
      }
      // Chain ends when we couldn't find an unmapped pred.
      if (PI == PE) break;
    }
  }
}

ReversePostOrderCFGView *
ReversePostOrderCFGView::create(AnalysisDeclContext &ctx) {
  const CFG *cfg = ctx.getCFG();
  if (!cfg)
    return nullptr;
  return new ReversePostOrderCFGView(cfg);
}
