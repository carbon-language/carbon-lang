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
#include "BinaryFunction.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include <limits>
#include <string>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

namespace llvm {
namespace bolt {

bool operator<(const BinaryBasicBlock &LHS, const BinaryBasicBlock &RHS) {
  return LHS.Index < RHS.Index;
}

MCInst *BinaryBasicBlock::getFirstNonPseudo() {
  auto &BC = Function->getBinaryContext();
  for (auto &Inst : Instructions) {
    if (!BC.MII->get(Inst.getOpcode()).isPseudo())
      return &Inst;
  }
  return nullptr;
}

MCInst *BinaryBasicBlock::getLastNonPseudo() {
  auto &BC = Function->getBinaryContext();
  for (auto Itr = Instructions.rbegin(); Itr != Instructions.rend(); ++Itr) {
    if (!BC.MII->get(Itr->getOpcode()).isPseudo())
      return &*Itr;
  }
  return nullptr;
}

BinaryBasicBlock *BinaryBasicBlock::getSuccessor(const MCSymbol *Label) const {
  if (!Label && succ_size() == 1)
    return *succ_begin();

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

void BinaryBasicBlock::replaceSuccessor(BinaryBasicBlock *Succ,
                                        BinaryBasicBlock *NewSucc,
                                        uint64_t Count,
                                        uint64_t MispredictedCount) {
  auto I = succ_begin();
  auto BI = BranchInfo.begin();
  for (; I != succ_end(); ++I) {
    assert(BI != BranchInfo.end() && "missing BranchInfo entry");
    if (*I == Succ)
      break;
    ++BI;
  }
  assert(I != succ_end() && "no such successor!");

  *I = NewSucc;
  *BI = BinaryBranchInfo{Count, MispredictedCount};
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

void BinaryBasicBlock::clearLandingPads() {
  for (auto *LPBlock : LandingPads) {
    auto count = LPBlock->Throwers.erase(this);
    assert(count == 1);
  }
  LandingPads.clear();
}

bool BinaryBasicBlock::analyzeBranch(const MCSymbol *&TBB,
                                     const MCSymbol *&FBB,
                                     MCInst *&CondBranch,
                                     MCInst *&UncondBranch) {
  auto &MIA = Function->getBinaryContext().MIA;
  return MIA->analyzeBranch(Instructions, TBB, FBB, CondBranch, UncondBranch);
}

bool BinaryBasicBlock::swapConditionalSuccessors() {
  if (succ_size() != 2)
    return false;

  std::swap(Successors[0], Successors[1]);
  std::swap(BranchInfo[0], BranchInfo[1]);
  return true;
}

void BinaryBasicBlock::addBranchInstruction(const BinaryBasicBlock *Successor) {
  assert(isSuccessor(Successor));
  auto &BC = Function->getBinaryContext();
  MCInst NewInst;
  BC.MIA->createUncondBranch(NewInst, Successor->getLabel(), BC.Ctx.get());
  Instructions.emplace_back(std::move(NewInst));
}

void BinaryBasicBlock::addTailCallInstruction(const MCSymbol *Target) {
  auto &BC = Function->getBinaryContext();
  MCInst NewInst;
  BC.MIA->createTailCall(NewInst, Target, BC.Ctx.get());
  Instructions.emplace_back(std::move(NewInst));
}

uint32_t BinaryBasicBlock::getNumPseudos() const {
#ifndef NDEBUG
  auto &BC = Function->getBinaryContext();
  uint32_t N = 0;
  for (auto &Instr : Instructions) {
    if (BC.MII->get(Instr.getOpcode()).isPseudo())
      ++N;
  }
  if (N != NumPseudos) {
    errs() << "BOLT-ERROR: instructions for basic block " << getName()
           << " in function " << *Function << ": calculated pseudos "
           << N << ", set pseudos " << NumPseudos << ", size " << size()
           << '\n';
    llvm_unreachable("pseudos mismatch");
  }
#endif
  return NumPseudos;
}

void BinaryBasicBlock::dump() const {
  auto &BC = Function->getBinaryContext();
  if (Label) outs() << Label->getName() << ":\n";
  BC.printInstructions(outs(), Instructions.begin(), Instructions.end(), Offset);
  outs() << "preds:";
  for (auto itr = pred_begin(); itr != pred_end(); ++itr) {
    outs() << " " << (*itr)->getName();
  }
  outs() << "\nsuccs:";
  for (auto itr = succ_begin(); itr != succ_end(); ++itr) {
    outs() << " " << (*itr)->getName();
  }
  outs() << "\n";
}

} // namespace bolt
} // namespace llvm
