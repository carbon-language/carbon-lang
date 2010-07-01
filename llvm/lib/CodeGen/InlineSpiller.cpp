//===-------- InlineSpiller.cpp - Insert spills and restores inline -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The inline spiller modifies the machine function directly instead of
// inserting spills and restores in VirtRegMap.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "spiller"
#include "Spiller.h"
#include "VirtRegMap.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
class InlineSpiller : public Spiller {
  MachineFunction &mf_;
  LiveIntervals &lis_;
  VirtRegMap &vrm_;
  MachineFrameInfo &mfi_;
  MachineRegisterInfo &mri_;
  const TargetInstrInfo &tii_;
  const TargetRegisterInfo &tri_;
  const BitVector reserved_;

  // Variables that are valid during spill(), but used by multiple methods.
  LiveInterval *li_;
  const TargetRegisterClass *rc_;
  int stackSlot_;
  const SmallVectorImpl<LiveInterval*> *spillIs_;

  ~InlineSpiller() {}

public:
  InlineSpiller(MachineFunction *mf, LiveIntervals *lis, VirtRegMap *vrm)
    : mf_(*mf), lis_(*lis), vrm_(*vrm),
      mfi_(*mf->getFrameInfo()),
      mri_(mf->getRegInfo()),
      tii_(*mf->getTarget().getInstrInfo()),
      tri_(*mf->getTarget().getRegisterInfo()),
      reserved_(tri_.getReservedRegs(mf_)) {}

  void spill(LiveInterval *li,
             std::vector<LiveInterval*> &newIntervals,
             SmallVectorImpl<LiveInterval*> &spillIs,
             SlotIndex *earliestIndex);
  bool reMaterialize(LiveInterval &NewLI, MachineBasicBlock::iterator MI);
  bool foldMemoryOperand(MachineBasicBlock::iterator MI,
                         const SmallVectorImpl<unsigned> &Ops);
  void insertReload(LiveInterval &NewLI, MachineBasicBlock::iterator MI);
  void insertSpill(LiveInterval &NewLI, MachineBasicBlock::iterator MI);
};
}

namespace llvm {
Spiller *createInlineSpiller(MachineFunction *mf,
                             LiveIntervals *lis,
                             const MachineLoopInfo *mli,
                             VirtRegMap *vrm) {
  return new InlineSpiller(mf, lis, vrm);
}
}

/// reMaterialize - Attempt to rematerialize li_->reg before MI instead of
/// reloading it.
bool InlineSpiller::reMaterialize(LiveInterval &NewLI,
                                  MachineBasicBlock::iterator MI) {
  SlotIndex UseIdx = lis_.getInstructionIndex(MI).getUseIndex();
  LiveRange *LR = li_->getLiveRangeContaining(UseIdx);
  if (!LR) {
    DEBUG(dbgs() << "\tundef at " << UseIdx << ", adding <undef> flags.\n");
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.isUse() && MO.getReg() == li_->reg)
        MO.setIsUndef();
    }
    return true;
  }

  // Find the instruction that defined this value of li_->reg.
  if (!LR->valno->isDefAccurate())
    return false;
  SlotIndex OrigDefIdx = LR->valno->def;
  MachineInstr *OrigDefMI = lis_.getInstructionFromIndex(OrigDefIdx);
  if (!OrigDefMI)
    return false;

  // FIXME: Provide AliasAnalysis argument.
  if (!tii_.isTriviallyReMaterializable(OrigDefMI))
    return false;

  // A rematerializable instruction may be using other virtual registers.
  // Make sure they are available at the new location.
  for (unsigned i = 0, e = OrigDefMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = OrigDefMI->getOperand(i);
    if (!MO.isReg() || !MO.getReg() || MO.getReg() == li_->reg)
      continue;
    // Reserved physregs are OK. Others are not (probably from coalescing).
    if (TargetRegisterInfo::isPhysicalRegister(MO.getReg())) {
      if (reserved_.test(MO.getReg()))
        continue;
      else
        return false;
    }
    // We don't want to move any virtual defs.
    if (MO.isDef())
      return false;
    // We have a use of a virtual register other than li_->reg.
    if (MO.isUndef())
      continue;
    // We cannot depend on virtual registers in spillIs_. They will be spilled.
    for (unsigned si = 0, se = spillIs_->size(); si != se; ++si)
      if ((*spillIs_)[si]->reg == MO.getReg())
        return false;

    // Is the register available here with the same value as at OrigDefMI?
    LiveInterval &ULI = lis_.getInterval(MO.getReg());
    LiveRange *HereLR = ULI.getLiveRangeContaining(UseIdx);
    if (!HereLR)
      return false;
    LiveRange *DefLR = ULI.getLiveRangeContaining(OrigDefIdx.getUseIndex());
    if (!DefLR || DefLR->valno != HereLR->valno)
      return false;
  }

  // Finally we can rematerialize OrigDefMI before MI.
  MachineBasicBlock &MBB = *MI->getParent();
  tii_.reMaterialize(MBB, MI, NewLI.reg, 0, OrigDefMI, tri_);
  SlotIndex DefIdx = lis_.InsertMachineInstrInMaps(--MI).getDefIndex();
  DEBUG(dbgs() << "\tremat:  " << DefIdx << '\t' << *MI);
  VNInfo *DefVNI = NewLI.getNextValue(DefIdx, 0, true,
                                       lis_.getVNInfoAllocator());
  NewLI.addRange(LiveRange(DefIdx, UseIdx.getDefIndex(), DefVNI));
  return true;
}

/// foldMemoryOperand - Try folding stack slot references in Ops into MI.
/// Return true on success, and MI will be erased.
bool InlineSpiller::foldMemoryOperand(MachineBasicBlock::iterator MI,
                                      const SmallVectorImpl<unsigned> &Ops) {
  // TargetInstrInfo::foldMemoryOperand only expects explicit, non-tied
  // operands.
  SmallVector<unsigned, 8> FoldOps;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    unsigned Idx = Ops[i];
    MachineOperand &MO = MI->getOperand(Idx);
    if (MO.isImplicit())
      continue;
    // FIXME: Teach targets to deal with subregs.
    if (MO.getSubReg())
      return false;
    // Tied use operands should not be passed to foldMemoryOperand.
    if (!MI->isRegTiedToDefOperand(Idx))
      FoldOps.push_back(Idx);
  }

  MachineInstr *FoldMI = tii_.foldMemoryOperand(mf_, MI, FoldOps, stackSlot_);
  if (!FoldMI)
    return false;
  MachineBasicBlock &MBB = *MI->getParent();
  lis_.ReplaceMachineInstrInMaps(MI, FoldMI);
  vrm_.addSpillSlotUse(stackSlot_, FoldMI);
  MBB.insert(MBB.erase(MI), FoldMI);
  DEBUG(dbgs() << "\tfolded: " << *FoldMI);
  return true;
}

/// insertReload - Insert a reload of NewLI.reg before MI.
void InlineSpiller::insertReload(LiveInterval &NewLI,
                                 MachineBasicBlock::iterator MI) {
  MachineBasicBlock &MBB = *MI->getParent();
  SlotIndex Idx = lis_.getInstructionIndex(MI).getDefIndex();
  tii_.loadRegFromStackSlot(MBB, MI, NewLI.reg, stackSlot_, rc_, &tri_);
  --MI; // Point to load instruction.
  SlotIndex LoadIdx = lis_.InsertMachineInstrInMaps(MI).getDefIndex();
  vrm_.addSpillSlotUse(stackSlot_, MI);
  DEBUG(dbgs() << "\treload:  " << LoadIdx << '\t' << *MI);
  VNInfo *LoadVNI = NewLI.getNextValue(LoadIdx, 0, true,
                                       lis_.getVNInfoAllocator());
  NewLI.addRange(LiveRange(LoadIdx, Idx, LoadVNI));
}

/// insertSpill - Insert a spill of NewLI.reg after MI.
void InlineSpiller::insertSpill(LiveInterval &NewLI,
                                MachineBasicBlock::iterator MI) {
  MachineBasicBlock &MBB = *MI->getParent();
  SlotIndex Idx = lis_.getInstructionIndex(MI).getDefIndex();
  tii_.storeRegToStackSlot(MBB, ++MI, NewLI.reg, true, stackSlot_, rc_, &tri_);
  --MI; // Point to store instruction.
  SlotIndex StoreIdx = lis_.InsertMachineInstrInMaps(MI).getDefIndex();
  vrm_.addSpillSlotUse(stackSlot_, MI);
  DEBUG(dbgs() << "\tspilled: " << StoreIdx << '\t' << *MI);
  VNInfo *StoreVNI = NewLI.getNextValue(Idx, 0, true,
                                        lis_.getVNInfoAllocator());
  NewLI.addRange(LiveRange(Idx, StoreIdx, StoreVNI));
}

void InlineSpiller::spill(LiveInterval *li,
                          std::vector<LiveInterval*> &newIntervals,
                          SmallVectorImpl<LiveInterval*> &spillIs,
                          SlotIndex *earliestIndex) {
  DEBUG(dbgs() << "Inline spilling " << *li << "\n");
  assert(li->isSpillable() && "Attempting to spill already spilled value.");
  assert(!li->isStackSlot() && "Trying to spill a stack slot.");

  li_ = li;
  rc_ = mri_.getRegClass(li->reg);
  stackSlot_ = vrm_.assignVirt2StackSlot(li->reg);
  spillIs_ = &spillIs;

  // Iterate over instructions using register.
  for (MachineRegisterInfo::reg_iterator RI = mri_.reg_begin(li->reg);
       MachineInstr *MI = RI.skipInstruction();) {

    // Analyze instruction.
    bool Reads, Writes;
    SmallVector<unsigned, 8> Ops;
    tie(Reads, Writes) = MI->readsWritesVirtualRegister(li->reg, &Ops);

    // Allocate interval around instruction.
    // FIXME: Infer regclass from instruction alone.
    unsigned NewVReg = mri_.createVirtualRegister(rc_);
    vrm_.grow();
    LiveInterval &NewLI = lis_.getOrCreateInterval(NewVReg);
    NewLI.markNotSpillable();

    // Attempt remat instead of reload.
    bool NeedsReload = Reads && !reMaterialize(NewLI, MI);

    // Attempt to fold memory ops.
    if (NewLI.empty() && foldMemoryOperand(MI, Ops))
      continue;

    if (NeedsReload)
      insertReload(NewLI, MI);

    // Rewrite instruction operands.
    bool hasLiveDef = false;
    for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(Ops[i]);
      MO.setReg(NewVReg);
      if (MO.isUse()) {
        if (!MI->isRegTiedToDefOperand(Ops[i]))
          MO.setIsKill();
      } else {
        if (!MO.isDead())
          hasLiveDef = true;
      }
    }

    // FIXME: Use a second vreg if instruction has no tied ops.
    if (Writes && hasLiveDef)
      insertSpill(NewLI, MI);

    DEBUG(dbgs() << "\tinterval: " << NewLI << '\n');
    newIntervals.push_back(&NewLI);
  }
}
