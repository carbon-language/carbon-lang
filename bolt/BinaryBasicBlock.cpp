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
#include "BinaryContext.h"
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

BinaryBasicBlock *BinaryBasicBlock::getSuccessor(const MCSymbol *Label) const {
  for (BinaryBasicBlock *BB : successors()) {
    if (BB->getLabel() == Label)
      return BB;
  }

  return nullptr;
}

BinaryBasicBlock *BinaryBasicBlock::getLandingPad(const MCSymbol *Label) const {
  for (BinaryBasicBlock *BB : landing_pads()) {
    if (BB->getLabel() == Label)
      return BB;
  }

  return nullptr;
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

void BinaryBasicBlock::addLandingPad(BinaryBasicBlock *LPBlock) {
  LandingPads.insert(LPBlock);
  LPBlock->Throwers.insert(this);
}

bool BinaryBasicBlock::analyzeBranch(const MCInstrAnalysis &MIA,
                                     const MCSymbol *&TBB,
                                     const MCSymbol *&FBB,
                                     MCInst *&CondBranch,
                                     MCInst *&UncondBranch) {
  return MIA.analyzeBranch(Instructions, TBB, FBB, CondBranch, UncondBranch);
}

void BinaryBasicBlock::dump(BinaryContext& BC) const {
  if (Label) dbgs() << Label->getName() << ":\n";
  BC.printInstructions(dbgs(), Instructions.begin(), Instructions.end(), Offset);
  dbgs() << "preds:";
  for (auto itr = pred_begin(); itr != pred_end(); ++itr) {
    dbgs() << " " << (*itr)->getName();
  }
  dbgs() << "\nsuccs:";
  for (auto itr = succ_begin(); itr != succ_end(); ++itr) {
    dbgs() << " " << (*itr)->getName();
  }
  dbgs() << "\n";
}

} // namespace bolt
} // namespace llvm
