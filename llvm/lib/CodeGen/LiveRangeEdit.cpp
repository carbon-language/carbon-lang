//===--- LiveRangeEdit.cpp - Basic tools for editing a register live range --===//
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

#include "LiveRangeEdit.h"
#include "VirtRegMap.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"

using namespace llvm;

LiveInterval &LiveRangeEdit::create(MachineRegisterInfo &mri,
                                    LiveIntervals &lis,
                                    VirtRegMap &vrm) {
  const TargetRegisterClass *RC = mri.getRegClass(parent_.reg);
  unsigned VReg = mri.createVirtualRegister(RC);
  vrm.grow();
  LiveInterval &li = lis.getOrCreateInterval(VReg);
  newRegs_.push_back(&li);
  return li;
}

void LiveRangeEdit::scanRemattable(LiveIntervals &lis,
                                   const TargetInstrInfo &tii,
                                   AliasAnalysis *aa) {
  for (LiveInterval::vni_iterator I = parent_.vni_begin(),
       E = parent_.vni_end(); I != E; ++I) {
    VNInfo *VNI = *I;
    if (VNI->isUnused())
      continue;
    MachineInstr *DefMI = lis.getInstructionFromIndex(VNI->def);
    if (!DefMI)
      continue;
    if (tii.isTriviallyReMaterializable(DefMI, aa))
      remattable_.insert(VNI);
  }
  scannedRemattable_ = true;
}

bool LiveRangeEdit::anyRematerializable(LiveIntervals &lis,
                                        const TargetInstrInfo &tii,
                                        AliasAnalysis *aa) {
  if (!scannedRemattable_)
    scanRemattable(lis, tii, aa);
  return !remattable_.empty();
}

/// allUsesAvailableAt - Return true if all registers used by OrigMI at
/// OrigIdx are also available with the same value at UseIdx.
bool LiveRangeEdit::allUsesAvailableAt(const MachineInstr *OrigMI,
                                       SlotIndex OrigIdx,
                                       SlotIndex UseIdx,
                                       LiveIntervals &lis) {
  OrigIdx = OrigIdx.getUseIndex();
  UseIdx = UseIdx.getUseIndex();
  for (unsigned i = 0, e = OrigMI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = OrigMI->getOperand(i);
    if (!MO.isReg() || !MO.getReg() || MO.getReg() == getReg())
      continue;
    // Reserved registers are OK.
    if (MO.isUndef() || !lis.hasInterval(MO.getReg()))
      continue;
    // We don't want to move any defs.
    if (MO.isDef())
      return false;
    // We cannot depend on virtual registers in uselessRegs_.
    for (unsigned ui = 0, ue = uselessRegs_.size(); ui != ue; ++ui)
      if (uselessRegs_[ui]->reg == MO.getReg())
        return false;

    LiveInterval &li = lis.getInterval(MO.getReg());
    const VNInfo *OVNI = li.getVNInfoAt(OrigIdx);
    if (!OVNI)
      continue;
    if (OVNI != li.getVNInfoAt(UseIdx))
      return false;
  }
  return true;
}

LiveRangeEdit::Remat LiveRangeEdit::canRematerializeAt(VNInfo *ParentVNI,
                                                       SlotIndex UseIdx,
                                                       bool cheapAsAMove,
                                                       LiveIntervals &lis) {
  assert(scannedRemattable_ && "Call anyRematerializable first");
  Remat RM = { 0, 0 };

  // We could remat an undefined value as IMPLICIT_DEF, but all that should have
  // been taken care of earlier.
  if (!(RM.ParentVNI = parent_.getVNInfoAt(UseIdx)))
    return RM;

  // Use scanRemattable info.
  if (!remattable_.count(RM.ParentVNI))
    return RM;

  // No defining instruction.
  MachineInstr *OrigMI = lis.getInstructionFromIndex(RM.ParentVNI->def);
  assert(OrigMI && "Defining instruction for remattable value disappeared");

  // If only cheap remats were requested, bail out early.
  if (cheapAsAMove && !OrigMI->getDesc().isAsCheapAsAMove())
    return RM;

  // Verify that all used registers are available with the same values.
  if (!allUsesAvailableAt(OrigMI, RM.ParentVNI->def, UseIdx, lis))
    return RM;

  RM.OrigMI = OrigMI;
  return RM;
}

SlotIndex LiveRangeEdit::rematerializeAt(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                         unsigned DestReg,
                                         const Remat &RM,
                                         LiveIntervals &lis,
                                         const TargetInstrInfo &tii,
                                         const TargetRegisterInfo &tri) {
  assert(RM.OrigMI && "Invalid remat");
  tii.reMaterialize(MBB, MI, DestReg, 0, RM.OrigMI, tri);
  rematted_.insert(RM.ParentVNI);
  return lis.InsertMachineInstrInMaps(--MI).getDefIndex();
}

