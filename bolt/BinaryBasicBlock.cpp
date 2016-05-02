//===--- BinaryBasicBlock.cpp - Interface for assembly-level basic block --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "BinaryBasicBlock.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include <limits>
#include <string>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

namespace llvm {
namespace bolt {

bool operator<(const BinaryBasicBlock &LHS, const BinaryBasicBlock &RHS) {
  return LHS.Offset < RHS.Offset;
}

void BinaryBasicBlock::addSuccessor(BinaryBasicBlock *Succ,
                                    uint64_t Count,
                                    uint64_t MispredictedCount) {
  Successors.push_back(Succ);
  BranchInfo.push_back({Count, MispredictedCount});
  Succ->Predecessors.push_back(this);
}

void BinaryBasicBlock::removeSuccessor(BinaryBasicBlock *Succ) {
  Succ->removePredecessor(this);
  auto I = succ_begin();
  auto BI = BranchInfo.begin();
  for (; I != succ_end(); ++I) {
    assert(BI != BranchInfo.end() && "missing BranchInfo entry");
    if (*I == Succ)
      break;
    ++BI;
  }
  assert(I != succ_end() && "no such successor!");

  Successors.erase(I);
  BranchInfo.erase(BI);
}

void BinaryBasicBlock::addPredecessor(BinaryBasicBlock *Pred) {
  Predecessors.push_back(Pred);
}

void BinaryBasicBlock::removePredecessor(BinaryBasicBlock *Pred) {
  auto I = std::find(pred_begin(), pred_end(), Pred);
  assert(I != pred_end() && "Pred is not a predecessor of this block!");
  Predecessors.erase(I);
}

bool BinaryBasicBlock::analyzeBranch(const MCInstrAnalysis &MIA,
                                     const MCSymbol *&TBB,
                                     const MCSymbol *&FBB,
                                     MCInst *&CondBranch,
                                     MCInst *&UncondBranch) {
  return MIA.analyzeBranch(Instructions, TBB, FBB, CondBranch, UncondBranch);
}

} // namespace bolt
} // namespace llvm
