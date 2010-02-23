//=- ReachableCodePathInsensitive.cpp ---------------------------*- C++ --*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a flow-sensitive, path-insensitive analysis of
// determining reachable blocks within a CFG.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/Analysis/Analyses/ReachableCode.h"
#include "clang/Analysis/CFG.h"

using namespace clang;

/// ScanReachableFromBlock - Mark all blocks reachable from Start.
/// Returns the total number of blocks that were marked reachable.
unsigned clang::ScanReachableFromBlock(const CFGBlock &Start,
                                       llvm::BitVector &Reachable) {
  unsigned count = 0;
  llvm::SmallVector<const CFGBlock*, 12> WL;
  
  // Prep work queue
  Reachable.set(Start.getBlockID());
  ++count;
  WL.push_back(&Start);
  
  // Find the reachable blocks from 'Start'.
  while (!WL.empty()) {
    const CFGBlock *item = WL.back();
    WL.pop_back();
    
    // Look at the successors and mark then reachable.
    for (CFGBlock::const_succ_iterator I=item->succ_begin(), E=item->succ_end();
         I != E; ++I)
      if (const CFGBlock *B = *I) {
        unsigned blockID = B->getBlockID();
        if (!Reachable[blockID]) {
          Reachable.set(blockID);
          ++count;
          WL.push_back(B);
        }
      }
  }
  return count;
}
