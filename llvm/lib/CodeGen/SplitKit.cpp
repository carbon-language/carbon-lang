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
#include "VirtRegMap.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

static cl::opt<bool>
AllowSplit("spiller-splits-edges",
           cl::desc("Allow critical edge splitting during spilling"));

//===----------------------------------------------------------------------===//
//                                 Split Analysis
//===----------------------------------------------------------------------===//

SplitAnalysis::SplitAnalysis(const MachineFunction &mf,
                             const LiveIntervals &lis,
                             const MachineLoopInfo &mli)
  : mf_(mf),
    lis_(lis),
    loops_(mli),
    tii_(*mf.getTarget().getInstrInfo()),
    curli_(0) {}

void SplitAnalysis::clear() {
  usingInstrs_.clear();
  usingBlocks_.clear();
  usingLoops_.clear();
  curli_ = 0;
}

bool SplitAnalysis::canAnalyzeBranch(const MachineBasicBlock *MBB) {
  MachineBasicBlock *T, *F;
  SmallVector<MachineOperand, 4> Cond;
  return !tii_.AnalyzeBranch(const_cast<MachineBasicBlock&>(*MBB), T, F, Cond);
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

// Get three sets of basic blocks surrounding a loop: Blocks inside the loop,
// predecessor blocks, and exit blocks.
void SplitAnalysis::getLoopBlocks(const MachineLoop *Loop, LoopBlocks &Blocks) {
  Blocks.clear();

  // Blocks in the loop.
  Blocks.Loop.insert(Loop->block_begin(), Loop->block_end());

  // Predecessor blocks.
  const MachineBasicBlock *Header = Loop->getHeader();
  for (MachineBasicBlock::const_pred_iterator I = Header->pred_begin(),
       E = Header->pred_end(); I != E; ++I)
    if (!Blocks.Loop.count(*I))
      Blocks.Preds.insert(*I);

  // Exit blocks.
  for (MachineLoop::block_iterator I = Loop->block_begin(),
       E = Loop->block_end(); I != E; ++I) {
    const MachineBasicBlock *MBB = *I;
    for (MachineBasicBlock::const_succ_iterator SI = MBB->succ_begin(),
       SE = MBB->succ_end(); SI != SE; ++SI)
      if (!Blocks.Loop.count(*SI))
        Blocks.Exits.insert(*SI);
  }
}

/// analyzeLoopPeripheralUse - Return an enum describing how curli_ is used in
/// and around the Loop.
SplitAnalysis::LoopPeripheralUse SplitAnalysis::
analyzeLoopPeripheralUse(const SplitAnalysis::LoopBlocks &Blocks) {
  LoopPeripheralUse use = ContainedInLoop;
  for (BlockCountMap::iterator I = usingBlocks_.begin(), E = usingBlocks_.end();
       I != E; ++I) {
    const MachineBasicBlock *MBB = I->first;
    // Is this a peripheral block?
    if (use < MultiPeripheral &&
        (Blocks.Preds.count(MBB) || Blocks.Exits.count(MBB))) {
      if (I->second > 1) use = MultiPeripheral;
      else               use = SinglePeripheral;
      continue;
    }
    // Is it a loop block?
    if (Blocks.Loop.count(MBB))
      continue;
    // It must be an unrelated block.
    return OutsideLoop;
  }
  return use;
}

/// getCriticalExits - It may be necessary to partially break critical edges
/// leaving the loop if an exit block has phi uses of curli. Collect the exit
/// blocks that need special treatment into CriticalExits.
void SplitAnalysis::getCriticalExits(const SplitAnalysis::LoopBlocks &Blocks,
                                     BlockPtrSet &CriticalExits) {
  CriticalExits.clear();

  // A critical exit block contains a phi def of curli, and has a predecessor
  // that is not in the loop nor a loop predecessor.
  // For such an exit block, the edges carrying the new variable must be moved
  // to a new pre-exit block.
  for (BlockPtrSet::iterator I = Blocks.Exits.begin(), E = Blocks.Exits.end();
       I != E; ++I) {
    const MachineBasicBlock *Succ = *I;
    SlotIndex SuccIdx = lis_.getMBBStartIdx(Succ);
    VNInfo *SuccVNI = curli_->getVNInfoAt(SuccIdx);
    // This exit may not have curli live in at all. No need to split.
    if (!SuccVNI)
      continue;
    // If this is not a PHI def, it is either using a value from before the
    // loop, or a value defined inside the loop. Both are safe.
    if (!SuccVNI->isPHIDef() || SuccVNI->def.getBaseIndex() != SuccIdx)
      continue;
    // This exit block does have a PHI. Does it also have a predecessor that is
    // not a loop block or loop predecessor?
    for (MachineBasicBlock::const_pred_iterator PI = Succ->pred_begin(),
         PE = Succ->pred_end(); PI != PE; ++PI) {
      const MachineBasicBlock *Pred = *PI;
      if (Blocks.Loop.count(Pred) || Blocks.Preds.count(Pred))
        continue;
      // This is a critical exit block, and we need to split the exit edge.
      CriticalExits.insert(Succ);
      break;
    }
  }
}

/// canSplitCriticalExits - Return true if it is possible to insert new exit
/// blocks before the blocks in CriticalExits.
bool
SplitAnalysis::canSplitCriticalExits(const SplitAnalysis::LoopBlocks &Blocks,
                                     BlockPtrSet &CriticalExits) {
  // If we don't allow critical edge splitting, require no critical exits.
  if (!AllowSplit)
    return CriticalExits.empty();

  for (BlockPtrSet::iterator I = CriticalExits.begin(), E = CriticalExits.end();
       I != E; ++I) {
    const MachineBasicBlock *Succ = *I;
    // We want to insert a new pre-exit MBB before Succ, and change all the
    // in-loop blocks to branch to the pre-exit instead of Succ.
    // Check that all the in-loop predecessors can be changed.
    for (MachineBasicBlock::const_pred_iterator PI = Succ->pred_begin(),
         PE = Succ->pred_end(); PI != PE; ++PI) {
      const MachineBasicBlock *Pred = *PI;
      // The external predecessors won't be altered.
      if (!Blocks.Loop.count(Pred) && !Blocks.Preds.count(Pred))
        continue;
      if (!canAnalyzeBranch(Pred))
        return false;
    }

    // If Succ's layout predecessor falls through, that too must be analyzable.
    // We need to insert the pre-exit block in the gap.
    MachineFunction::const_iterator MFI = Succ;
    if (MFI == mf_.begin())
      continue;
    if (!canAnalyzeBranch(--MFI))
      return false;
  }
  // No problems found.
  return true;
}

void SplitAnalysis::analyze(const LiveInterval *li) {
  clear();
  curli_ = li;
  analyzeUses();
}

const MachineLoop *SplitAnalysis::getBestSplitLoop() {
  assert(curli_ && "Call analyze() before getBestSplitLoop");
  if (usingLoops_.empty())
    return 0;

  LoopPtrSet Loops, SecondLoops;
  LoopBlocks Blocks;
  BlockPtrSet CriticalExits;

  // Find first-class and second class candidate loops.
  // We prefer to split around loops where curli is used outside the periphery.
  for (LoopPtrSet::const_iterator I = usingLoops_.begin(),
       E = usingLoops_.end(); I != E; ++I) {
    getLoopBlocks(*I, Blocks);
    LoopPtrSet *LPS = 0;
    switch(analyzeLoopPeripheralUse(Blocks)) {
    case OutsideLoop:
      LPS = &Loops;
      break;
    case MultiPeripheral:
      LPS = &SecondLoops;
      break;
    case ContainedInLoop:
      DEBUG(dbgs() << "ContainedInLoop: " << **I);
      continue;
    case SinglePeripheral:
      DEBUG(dbgs() << "SinglePeripheral: " << **I);
      continue;
    }
    // Will it be possible to split around this loop?
    getCriticalExits(Blocks, CriticalExits);
    DEBUG(dbgs() << CriticalExits.size() << " critical exits: " << **I);
    if (!canSplitCriticalExits(Blocks, CriticalExits))
      continue;
    // This is a possible split.
    assert(LPS);
    LPS->insert(*I);
  }

  DEBUG(dbgs() << "Got " << Loops.size() << " + " << SecondLoops.size()
               << " candidate loops\n");

  // If there are no first class loops available, look at second class loops.
  if (Loops.empty())
    Loops = SecondLoops;

  if (Loops.empty())
    return 0;

  // Pick the earliest loop.
  // FIXME: Are there other heuristics to consider?
  const MachineLoop *Best = 0;
  SlotIndex BestIdx;
  for (LoopPtrSet::const_iterator I = Loops.begin(), E = Loops.end(); I != E;
       ++I) {
    SlotIndex Idx = lis_.getMBBStartIdx((*I)->getHeader());
    if (!Best || Idx < BestIdx)
      Best = *I, BestIdx = Idx;
  }
  DEBUG(dbgs() << "Best: " << *Best);
  return Best;
}


//===----------------------------------------------------------------------===//
//                               Split Editor
//===----------------------------------------------------------------------===//

/// Create a new SplitEditor for editing the LiveInterval analyzed by SA.
SplitEditor::SplitEditor(SplitAnalysis &sa, LiveIntervals &lis, VirtRegMap &vrm)
  : sa_(sa), lis_(lis), vrm_(vrm),
    mri_(vrm.getMachineFunction().getRegInfo()),
    tii_(*vrm.getMachineFunction().getTarget().getInstrInfo()),
    dupli_(0), openli_(0)
{
  const LiveInterval *curli = sa_.getCurLI();
  assert(curli && "SplitEditor created from empty SplitAnalysis");

  // Make sure curli is assigned a stack slot, so all our intervals get the
  // same slot as curli.
  if (vrm_.getStackSlot(curli->reg) == VirtRegMap::NO_STACK_SLOT)
    vrm_.assignVirt2StackSlot(curli->reg);

  // Create an interval for dupli that is a copy of curli.
  dupli_ = createInterval();
  dupli_->Copy(*curli, &mri_, lis_.getVNInfoAllocator());
  DEBUG(dbgs() << "SplitEditor DupLI: " << *dupli_ << '\n');
}

LiveInterval *SplitEditor::createInterval() {
  unsigned curli = sa_.getCurLI()->reg;
  unsigned Reg = mri_.createVirtualRegister(mri_.getRegClass(curli));
  LiveInterval &Intv = lis_.getOrCreateInterval(Reg);
  vrm_.grow();
  vrm_.assignVirt2StackSlot(Reg, vrm_.getStackSlot(curli));
  return &Intv;
}

VNInfo *SplitEditor::mapValue(VNInfo *dupliVNI) {
  VNInfo *&VNI = valueMap_[dupliVNI];
  if (!VNI)
    VNI = openli_->createValueCopy(dupliVNI, lis_.getVNInfoAllocator());
  return VNI;
}

/// Create a new virtual register and live interval to be used by following
/// use* and copy* calls.
void SplitEditor::openLI() {
  assert(!openli_ && "Previous LI not closed before openLI");
  openli_ = createInterval();
}

/// copyToPHI - Insert a copy to openli at the end of A, and catch it with a
/// PHI def at the beginning of the successor B. This call is ignored if dupli
/// is not live out of A.
void SplitEditor::copyToPHI(MachineBasicBlock &A, MachineBasicBlock &B) {
  assert(openli_ && "openLI not called before copyToPHI");

  SlotIndex EndA = lis_.getMBBEndIdx(&A);
  VNInfo *DupVNIA = dupli_->getVNInfoAt(EndA.getPrevIndex());
  if (!DupVNIA) {
    DEBUG(dbgs() << "  ignoring copyToPHI, dupli not live out of BB#"
                 << A.getNumber() << ".\n");
    return;
  }

  // Insert the COPY instruction at the end of A.
  MachineInstr *MI = BuildMI(A, A.getFirstTerminator(), DebugLoc(),
                                 tii_.get(TargetOpcode::COPY), dupli_->reg)
                           .addReg(openli_->reg);
  SlotIndex DefIdx = lis_.InsertMachineInstrInMaps(MI).getDefIndex();

  // Add a phi kill value and live range out of A.
  VNInfo *VNIA = openli_->getNextValue(DefIdx, MI, true,
                                       lis_.getVNInfoAllocator());
  openli_->addRange(LiveRange(DefIdx, EndA, VNIA));

  // Now look at the start of B.
  SlotIndex StartB = lis_.getMBBStartIdx(&B);
  SlotIndex EndB = lis_.getMBBEndIdx(&B);
  LiveRange *DupB = dupli_->getLiveRangeContaining(StartB);
  if (!DupB) {
    DEBUG(dbgs() << "  copyToPHI:, dupli not live in to BB#"
                 << B.getNumber() << ".\n");
    return;
  }

  VNInfo *VNIB = openli_->getVNInfoAt(StartB);
  if (!VNIB) {
    // Create a phi value.
    VNIB = openli_->getNextValue(SlotIndex(StartB, true), 0, false,
                                 lis_.getVNInfoAllocator());
    VNIB->setIsPHIDef(true);
    // Add a minimal range for the new value.
    openli_->addRange(LiveRange(VNIB->def, std::min(EndB, DupB->end), VNIB));

    VNInfo *&mapVNI = valueMap_[DupB->valno];
    if (mapVNI) {
      // Multiple copies - must create PHI value.
      abort();
    } else {
      // This is the first copy of dupLR. Mark the mapping.
      mapVNI = VNIB;
    }

  }

  DEBUG(dbgs() << "  copyToPHI at " << DefIdx << ": " << *openli_ << '\n');
}

/// useLI - indicate that all instructions in MBB should use openli.
void SplitEditor::useLI(const MachineBasicBlock &MBB) {
  useLI(lis_.getMBBStartIdx(&MBB), lis_.getMBBEndIdx(&MBB));
}

void SplitEditor::useLI(SlotIndex Start, SlotIndex End) {
  assert(openli_ && "openLI not called before useLI");

  // Map the dupli values from the interval into openli_
  LiveInterval::const_iterator B = dupli_->begin(), E = dupli_->end();
  LiveInterval::const_iterator I = std::lower_bound(B, E, Start);

  if (I != B) {
    --I;
    // I begins before Start, but overlaps. openli may already have a value from
    // copyToLI.
    if (I->end > Start && !openli_->liveAt(Start))
      openli_->addRange(LiveRange(Start, std::min(End, I->end),
                        mapValue(I->valno)));
    ++I;
  }

  // The remaining ranges begin after Start.
  for (;I != E && I->start < End; ++I)
    openli_->addRange(LiveRange(I->start, std::min(End, I->end),
                                mapValue(I->valno)));
  DEBUG(dbgs() << "  added range [" << Start << ';' << End << "): " << *openli_
               << '\n');
}

/// copyFromLI - Insert a copy back to dupli from openli at position I.
SlotIndex SplitEditor::copyFromLI(MachineBasicBlock &MBB, MachineBasicBlock::iterator I) {
  assert(openli_ && "openLI not called before copyFromLI");

  // Insert the COPY instruction.
  MachineInstr *MI =
    BuildMI(MBB, I, DebugLoc(), tii_.get(TargetOpcode::COPY), openli_->reg)
      .addReg(dupli_->reg);
  SlotIndex Idx = lis_.InsertMachineInstrInMaps(MI);

  DEBUG(dbgs() << "  copyFromLI at " << Idx << ": " << *openli_ << '\n');
  return Idx;
}

/// closeLI - Indicate that we are done editing the currently open
/// LiveInterval, and ranges can be trimmed.
void SplitEditor::closeLI() {
  assert(openli_ && "openLI not called before closeLI");
  openli_ = 0;
}

/// rewrite - after all the new live ranges have been created, rewrite
/// instructions using curli to use the new intervals.
void SplitEditor::rewrite() {
  assert(!openli_ && "Previous LI not closed before rewrite");
}


//===----------------------------------------------------------------------===//
//                               Loop Splitting
//===----------------------------------------------------------------------===//

void SplitEditor::splitAroundLoop(const MachineLoop *Loop) {
  SplitAnalysis::LoopBlocks Blocks;
  sa_.getLoopBlocks(Loop, Blocks);

  // Break critical edges as needed.
  SplitAnalysis::BlockPtrSet CriticalExits;
  sa_.getCriticalExits(Blocks, CriticalExits);
  assert(CriticalExits.empty() && "Cannot break critical exits yet");

  // Create new live interval for the loop.
  openLI();

  // Insert copies in the predecessors.
  for (SplitAnalysis::BlockPtrSet::iterator I = Blocks.Preds.begin(),
       E = Blocks.Preds.end(); I != E; ++I) {
    MachineBasicBlock &MBB = const_cast<MachineBasicBlock&>(**I);
    copyToPHI(MBB, *Loop->getHeader());
  }

  // Switch all loop blocks.
  for (SplitAnalysis::BlockPtrSet::iterator I = Blocks.Loop.begin(),
       E = Blocks.Loop.end(); I != E; ++I)
     useLI(**I);

  // Insert back copies in the exit blocks.
  for (SplitAnalysis::BlockPtrSet::iterator I = Blocks.Exits.begin(),
       E = Blocks.Exits.end(); I != E; ++I) {
    MachineBasicBlock &MBB = const_cast<MachineBasicBlock&>(**I);
    SlotIndex Start = lis_.getMBBStartIdx(&MBB);
    VNInfo *VNI = sa_.getCurLI()->getVNInfoAt(Start);
    // Only insert a back copy if curli is live and is either a phi or a value
    // defined inside the loop.
    if (!VNI) continue;
    if (openli_->liveAt(VNI->def) ||
        (VNI->isPHIDef() && VNI->def.getBaseIndex() == Start))
      copyFromLI(MBB, MBB.begin());
  }

  // Done.
  closeLI();
  rewrite();
  abort();
}

