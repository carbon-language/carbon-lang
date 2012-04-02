//===-- LiveRangeEdit.cpp - Basic tools for editing a register live range -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The LiveRangeEdit class represents changes done to a virtual register when it
// is spilled or split.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "LiveRangeEdit.h"
#include "VirtRegMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

STATISTIC(NumDCEDeleted,     "Number of instructions deleted by DCE");
STATISTIC(NumDCEFoldedLoads, "Number of single use loads folded after DCE");
STATISTIC(NumFracRanges,     "Number of live ranges fractured by DCE");

void LiveRangeEdit::Delegate::anchor() { }

LiveInterval &LiveRangeEdit::createFrom(unsigned OldReg) {
  unsigned VReg = MRI.createVirtualRegister(MRI.getRegClass(OldReg));
  VRM->grow();
  VRM->setIsSplitFromReg(VReg, VRM->getOriginal(OldReg));
  LiveInterval &LI = LIS.getOrCreateInterval(VReg);
  newRegs_.push_back(&LI);
  return LI;
}

bool LiveRangeEdit::checkRematerializable(VNInfo *VNI,
                                          const MachineInstr *DefMI,
                                          AliasAnalysis *aa) {
  assert(DefMI && "Missing instruction");
  scannedRemattable_ = true;
  if (!TII.isTriviallyReMaterializable(DefMI, aa))
    return false;
  remattable_.insert(VNI);
  return true;
}

void LiveRangeEdit::scanRemattable(AliasAnalysis *aa) {
  for (LiveInterval::vni_iterator I = parent_.vni_begin(),
       E = parent_.vni_end(); I != E; ++I) {
    VNInfo *VNI = *I;
    if (VNI->isUnused())
      continue;
    MachineInstr *DefMI = LIS.getInstructionFromIndex(VNI->def);
    if (!DefMI)
      continue;
    checkRematerializable(VNI, DefMI, aa);
  }
  scannedRemattable_ = true;
}

bool LiveRangeEdit::anyRematerializable(AliasAnalysis *aa) {
  if (!scannedRemattable_)
    scanRemattable(aa);
  return !remattable_.empty();
}

/// allUsesAvailableAt - Return true if all registers used by OrigMI at
/// OrigIdx are also available with the same value at UseIdx.
bool LiveRangeEdit::allUsesAvailableAt(const MachineInstr *OrigMI,
                                       SlotIndex OrigIdx,
                                       SlotIndex UseIdx) {
  OrigIdx = OrigIdx.getRegSlot(true);
  UseIdx = UseIdx.getRegSlot(true);
  for (unsigned i = 0, e = OrigMI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = OrigMI->getOperand(i);
    if (!MO.isReg() || !MO.getReg() || MO.isDef())
      continue;
    // Reserved registers are OK.
    if (MO.isUndef() || !LIS.hasInterval(MO.getReg()))
      continue;

    LiveInterval &li = LIS.getInterval(MO.getReg());
    const VNInfo *OVNI = li.getVNInfoAt(OrigIdx);
    if (!OVNI)
      continue;
    if (OVNI != li.getVNInfoAt(UseIdx))
      return false;
  }
  return true;
}

bool LiveRangeEdit::canRematerializeAt(Remat &RM,
                                       SlotIndex UseIdx,
                                       bool cheapAsAMove) {
  assert(scannedRemattable_ && "Call anyRematerializable first");

  // Use scanRemattable info.
  if (!remattable_.count(RM.ParentVNI))
    return false;

  // No defining instruction provided.
  SlotIndex DefIdx;
  if (RM.OrigMI)
    DefIdx = LIS.getInstructionIndex(RM.OrigMI);
  else {
    DefIdx = RM.ParentVNI->def;
    RM.OrigMI = LIS.getInstructionFromIndex(DefIdx);
    assert(RM.OrigMI && "No defining instruction for remattable value");
  }

  // If only cheap remats were requested, bail out early.
  if (cheapAsAMove && !RM.OrigMI->isAsCheapAsAMove())
    return false;

  // Verify that all used registers are available with the same values.
  if (!allUsesAvailableAt(RM.OrigMI, DefIdx, UseIdx))
    return false;

  return true;
}

SlotIndex LiveRangeEdit::rematerializeAt(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                         unsigned DestReg,
                                         const Remat &RM,
                                         const TargetRegisterInfo &tri,
                                         bool Late) {
  assert(RM.OrigMI && "Invalid remat");
  TII.reMaterialize(MBB, MI, DestReg, 0, RM.OrigMI, tri);
  rematted_.insert(RM.ParentVNI);
  return LIS.getSlotIndexes()->insertMachineInstrInMaps(--MI, Late)
           .getRegSlot();
}

void LiveRangeEdit::eraseVirtReg(unsigned Reg) {
  if (delegate_ && delegate_->LRE_CanEraseVirtReg(Reg))
    LIS.removeInterval(Reg);
}

bool LiveRangeEdit::foldAsLoad(LiveInterval *LI,
                               SmallVectorImpl<MachineInstr*> &Dead) {
  MachineInstr *DefMI = 0, *UseMI = 0;

  // Check that there is a single def and a single use.
  for (MachineRegisterInfo::reg_nodbg_iterator I = MRI.reg_nodbg_begin(LI->reg),
       E = MRI.reg_nodbg_end(); I != E; ++I) {
    MachineOperand &MO = I.getOperand();
    MachineInstr *MI = MO.getParent();
    if (MO.isDef()) {
      if (DefMI && DefMI != MI)
        return false;
      if (!MI->canFoldAsLoad())
        return false;
      DefMI = MI;
    } else if (!MO.isUndef()) {
      if (UseMI && UseMI != MI)
        return false;
      // FIXME: Targets don't know how to fold subreg uses.
      if (MO.getSubReg())
        return false;
      UseMI = MI;
    }
  }
  if (!DefMI || !UseMI)
    return false;

  DEBUG(dbgs() << "Try to fold single def: " << *DefMI
               << "       into single use: " << *UseMI);

  SmallVector<unsigned, 8> Ops;
  if (UseMI->readsWritesVirtualRegister(LI->reg, &Ops).second)
    return false;

  MachineInstr *FoldMI = TII.foldMemoryOperand(UseMI, Ops, DefMI);
  if (!FoldMI)
    return false;
  DEBUG(dbgs() << "                folded: " << *FoldMI);
  LIS.ReplaceMachineInstrInMaps(UseMI, FoldMI);
  UseMI->eraseFromParent();
  DefMI->addRegisterDead(LI->reg, 0);
  Dead.push_back(DefMI);
  ++NumDCEFoldedLoads;
  return true;
}

void LiveRangeEdit::eliminateDeadDefs(SmallVectorImpl<MachineInstr*> &Dead,
                                      ArrayRef<unsigned> RegsBeingSpilled) {
  SetVector<LiveInterval*,
            SmallVector<LiveInterval*, 8>,
            SmallPtrSet<LiveInterval*, 8> > ToShrink;

  for (;;) {
    // Erase all dead defs.
    while (!Dead.empty()) {
      MachineInstr *MI = Dead.pop_back_val();
      assert(MI->allDefsAreDead() && "Def isn't really dead");
      SlotIndex Idx = LIS.getInstructionIndex(MI).getRegSlot();

      // Never delete inline asm.
      if (MI->isInlineAsm()) {
        DEBUG(dbgs() << "Won't delete: " << Idx << '\t' << *MI);
        continue;
      }

      // Use the same criteria as DeadMachineInstructionElim.
      bool SawStore = false;
      if (!MI->isSafeToMove(&TII, 0, SawStore)) {
        DEBUG(dbgs() << "Can't delete: " << Idx << '\t' << *MI);
        continue;
      }

      DEBUG(dbgs() << "Deleting dead def " << Idx << '\t' << *MI);

      // Check for live intervals that may shrink
      for (MachineInstr::mop_iterator MOI = MI->operands_begin(),
             MOE = MI->operands_end(); MOI != MOE; ++MOI) {
        if (!MOI->isReg())
          continue;
        unsigned Reg = MOI->getReg();
        if (!TargetRegisterInfo::isVirtualRegister(Reg))
          continue;
        LiveInterval &LI = LIS.getInterval(Reg);

        // Shrink read registers, unless it is likely to be expensive and
        // unlikely to change anything. We typically don't want to shrink the
        // PIC base register that has lots of uses everywhere.
        // Always shrink COPY uses that probably come from live range splitting.
        if (MI->readsVirtualRegister(Reg) &&
            (MI->isCopy() || MOI->isDef() || MRI.hasOneNonDBGUse(Reg) ||
             LI.killedAt(Idx)))
          ToShrink.insert(&LI);

        // Remove defined value.
        if (MOI->isDef()) {
          if (VNInfo *VNI = LI.getVNInfoAt(Idx)) {
            if (delegate_)
              delegate_->LRE_WillShrinkVirtReg(LI.reg);
            LI.removeValNo(VNI);
            if (LI.empty()) {
              ToShrink.remove(&LI);
              eraseVirtReg(Reg);
            }
          }
        }
      }

      if (delegate_)
        delegate_->LRE_WillEraseInstruction(MI);
      LIS.RemoveMachineInstrFromMaps(MI);
      MI->eraseFromParent();
      ++NumDCEDeleted;
    }

    if (ToShrink.empty())
      break;

    // Shrink just one live interval. Then delete new dead defs.
    LiveInterval *LI = ToShrink.back();
    ToShrink.pop_back();
    if (foldAsLoad(LI, Dead))
      continue;
    if (delegate_)
      delegate_->LRE_WillShrinkVirtReg(LI->reg);
    if (!LIS.shrinkToUses(LI, &Dead))
      continue;
    if (!VRM)
      continue;
    
    // Don't create new intervals for a register being spilled.
    // The new intervals would have to be spilled anyway so its not worth it.
    // Also they currently aren't spilled so creating them and not spilling
    // them results in incorrect code.
    bool BeingSpilled = false;
    for (unsigned i = 0, e = RegsBeingSpilled.size(); i != e; ++i) {
      if (LI->reg == RegsBeingSpilled[i]) {
        BeingSpilled = true;
        break;
      }
    }
    
    if (BeingSpilled) continue;
    if (!VRM) continue;

    // LI may have been separated, create new intervals.
    LI->RenumberValues(LIS);
    ConnectedVNInfoEqClasses ConEQ(LIS);
    unsigned NumComp = ConEQ.Classify(LI);
    if (NumComp <= 1)
      continue;
    ++NumFracRanges;
    bool IsOriginal = VRM->getOriginal(LI->reg) == LI->reg;
    DEBUG(dbgs() << NumComp << " components: " << *LI << '\n');
    SmallVector<LiveInterval*, 8> Dups(1, LI);
    for (unsigned i = 1; i != NumComp; ++i) {
      Dups.push_back(&createFrom(LI->reg));
      // If LI is an original interval that hasn't been split yet, make the new
      // intervals their own originals instead of referring to LI. The original
      // interval must contain all the split products, and LI doesn't.
      if (IsOriginal)
        VRM->setIsSplitFromReg(Dups.back()->reg, 0);
      if (delegate_)
        delegate_->LRE_DidCloneVirtReg(Dups.back()->reg, LI->reg);
    }
    ConEQ.Distribute(&Dups[0], MRI);
  }
}

void LiveRangeEdit::calculateRegClassAndHint(MachineFunction &MF,
                                             const MachineLoopInfo &Loops) {
  VirtRegAuxInfo VRAI(MF, LIS, Loops);
  for (iterator I = begin(), E = end(); I != E; ++I) {
    LiveInterval &LI = **I;
    if (MRI.recomputeRegClass(LI.reg, MF.getTarget()))
      DEBUG(dbgs() << "Inflated " << PrintReg(LI.reg) << " to "
                   << MRI.getRegClass(LI.reg)->getName() << '\n');
    VRAI.CalculateWeightAndHint(LI);
  }
}
