//===---------- SplitKit.cpp - Toolkit for splitting live ranges ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SplitAnalysis class as well as mutator functions for
// live range splitting.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "splitter"
#include "SplitKit.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;


//===----------------------------------------------------------------------===//
//                                 Split Analysis
//===----------------------------------------------------------------------===//

SplitAnalysis::SplitAnalysis(const MachineFunction *mf,
                             const LiveIntervals *lis,
                             const MachineLoopInfo *mli)
  : mf_(*mf),
    lis_(*lis),
    loops_(*mli),
    curli_(0) {}

void SplitAnalysis::clear() {
  usingInstrs_.clear();
  usingBlocks_.clear();
  usingLoops_.clear();
}

/// analyzeUses - Count instructions, basic blocks, and loops using curli.
void SplitAnalysis::analyzeUses() {
  const MachineRegisterInfo &MRI = mf_.getRegInfo();
  for (MachineRegisterInfo::reg_iterator I = MRI.reg_begin(curli_->reg);
       MachineInstr *MI = I.skipInstruction();) {
    if (MI->isDebugValue() || !usingInstrs_.insert(MI))
      continue;
    MachineBasicBlock *MBB = MI->getParent();
    if (usingBlocks_[MBB]++)
      continue;
    if (MachineLoop *Loop = loops_.getLoopFor(MBB))
      usingLoops_.insert(Loop);
  }
  DEBUG(dbgs() << "Counted "
               << usingInstrs_.size() << " instrs, "
               << usingBlocks_.size() << " blocks, "
               << usingLoops_.size()  << " loops in "
               << *curli_ << "\n");
}

SplitAnalysis::LoopPeripheralUse
SplitAnalysis::analyzeLoopPeripheralUse(const MachineLoop *Loop) {
  // Peripheral blocks.
  SmallVector<MachineBasicBlock*, 16> Peri;
  Loop->getExitBlocks(Peri);
  if (MachineBasicBlock *PredBB = Loop->getLoopPredecessor())
    Peri.push_back(PredBB);
  array_pod_sort(Peri.begin(), Peri.end());
  Peri.erase(std::unique(Peri.begin(), Peri.end()), Peri.end());

  LoopPeripheralUse use = ContainedInLoop;
  for (BlockCountMap::iterator I = usingBlocks_.begin(), E = usingBlocks_.end();
       I != E; ++I) {
    const MachineBasicBlock *MBB = I->first;
    // Is this a peripheral block?
    if (use < MultiPeripheral &&
        std::binary_search(Peri.begin(), Peri.end(), MBB)) {
      if (I->second > 1) use = MultiPeripheral;
      else               use = SinglePeripheral;
      continue;
    }
    // Is it a loop block?
    if (Loop->contains(MBB))
      continue;
    // It must be an unrelated block.
    return OutsideLoop;
  }
  return use;
}

void SplitAnalysis::analyze(const LiveInterval *li) {
  clear();
  curli_ = li;
  analyzeUses();
}

const MachineLoop *SplitAnalysis::getBestSplitLoop() {
  LoopPtrSet Loops, SecondLoops;

  // Find first-class and second class candidate loops.
  // We prefer to split around loops where curli is used outside the periphery.
  for (LoopPtrSet::const_iterator I = usingLoops_.begin(),
       E = usingLoops_.end(); I != E; ++I)
    switch(analyzeLoopPeripheralUse(*I)) {
    case OutsideLoop:
      Loops.insert(*I);
      break;
    case MultiPeripheral:
      SecondLoops.insert(*I);
      break;
    default:
      continue;
    }

  // If there are no first class loops available, look at second class loops.
  if (Loops.empty())
    Loops = SecondLoops;

  if (Loops.empty())
    return 0;

  // Pick the earliest loop.
  // FIXME: Are there other heuristics to consider?
  // - avoid breaking critical edges.
  // - avoid impossible loops.
  const MachineLoop *Best = 0;
  SlotIndex BestIdx;
  for (LoopPtrSet::const_iterator I = Loops.begin(), E = Loops.end(); I != E;
       ++I) {
    SlotIndex Idx = lis_.getMBBStartIdx((*I)->getHeader());
    if (!Best || Idx < BestIdx)
      Best = *I, BestIdx = Idx;
  }
  return Best;
}

//===----------------------------------------------------------------------===//
//                               Loop Splitting
//===----------------------------------------------------------------------===//

bool llvm::splitAroundLoop(SplitAnalysis &sa, const MachineLoop *loop) {
  return false;
}

