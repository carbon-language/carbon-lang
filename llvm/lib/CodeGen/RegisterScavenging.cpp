//===-- RegisterScavenging.cpp - Machine register scavenging --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Evan Cheng and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the machine register scavenger. It can provide
// information such as unused register at any point in a machine basic block.
// It also provides a mechanism to make registers availbale by evicting them
// to spill slots.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reg-scavenging"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

void RegScavenger::enterBasicBlock(MachineBasicBlock *mbb) {
  const MachineFunction &MF = *mbb->getParent();
  const TargetMachine &TM = MF.getTarget();
  const MRegisterInfo *RegInfo = TM.getRegisterInfo();

  assert((NumPhysRegs == 0 || NumPhysRegs == RegInfo->getNumRegs()) &&
         "Target changed?");

  if (!MBB) {
    NumPhysRegs = RegInfo->getNumRegs();
    RegStates.resize(NumPhysRegs);

    // Create reserved registers bitvector.
    ReservedRegs = RegInfo->getReservedRegs(MF);

    // Create callee-saved registers bitvector.
    CalleeSavedRegs.resize(NumPhysRegs);
    const unsigned *CSRegs = RegInfo->getCalleeSavedRegs();
    if (CSRegs != NULL)
      for (unsigned i = 0; CSRegs[i]; ++i)
        CalleeSavedRegs.set(CSRegs[i]);
  }

  MBB = mbb;

  // All registers started out unused.
  RegStates.set();

  // Reserved registers are always used.
  RegStates ^= ReservedRegs;

  // Live-in registers are in use.
  if (!MBB->livein_empty())
    for (MachineBasicBlock::const_livein_iterator I = MBB->livein_begin(),
           E = MBB->livein_end(); I != E; ++I)
      setUsed(*I);

  Tracking = false;
}

void RegScavenger::forward() {
  // Move ptr forward.
  if (!Tracking) {
    MBBI = MBB->begin();
    Tracking = true;
  } else {
    assert(MBBI != MBB->end() && "Already at the end of the basic block!");
    MBBI = next(MBBI);
  }

  MachineInstr *MI = MBBI;
  // Process uses first.
  BitVector ChangedRegs(NumPhysRegs);
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isUse())
      continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0)
      continue;
    assert(isUsed(Reg));
    if (MO.isKill() && !isReserved(Reg))
      ChangedRegs.set(Reg);
  }
  // Change states of all registers after all the uses are processed to guard
  // against multiple uses.
  setUnused(ChangedRegs);

  // Process defs.
  const TargetInstrDescriptor *TID = MI->getInstrDescriptor();
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isDef())
      continue;
    unsigned Reg = MO.getReg();
    // Skip two-address destination operand.
    if (TID->findTiedToSrcOperand(i) != -1) {
      assert(isUsed(Reg));
      continue;
    }
    assert(isUnused(Reg) || isReserved(Reg));
    if (!MO.isDead())
      setUsed(Reg);
  }
}

void RegScavenger::backward() {
  assert(Tracking && "Not tracking states!");
  assert(MBBI != MBB->begin() && "Already at start of basic block!");
  // Move ptr backward.
  MBBI = prior(MBBI);

  MachineInstr *MI = MBBI;
  // Process defs first.
  const TargetInstrDescriptor *TID = MI->getInstrDescriptor();
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isDef())
      continue;
    // Skip two-address destination operand.
    if (TID->findTiedToSrcOperand(i) != -1)
      continue;
    unsigned Reg = MO.getReg();
    assert(isUsed(Reg));
    if (!isReserved(Reg))
      setUnused(Reg);
  }

  // Process uses.
  BitVector ChangedRegs(NumPhysRegs);
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isUse())
      continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0)
      continue;
    assert(isUnused(Reg) || isReserved(Reg));
    ChangedRegs.set(Reg);
  }
  setUsed(ChangedRegs);
}

/// CreateRegClassMask - Set the bits that represent the registers in the
/// TargetRegisterClass.
static void CreateRegClassMask(const TargetRegisterClass *RC, BitVector &Mask) {
  for (TargetRegisterClass::iterator I = RC->begin(), E = RC->end(); I != E;
       ++I)
    Mask.set(*I);
}

unsigned RegScavenger::FindUnusedReg(const TargetRegisterClass *RegClass,
                                     const BitVector &Candidates) const {
  // Mask off the registers which are not in the TargetRegisterClass.
  BitVector RegStatesCopy(NumPhysRegs, false);
  CreateRegClassMask(RegClass, RegStatesCopy);
  RegStatesCopy &= RegStates;

  // Restrict the search to candidates.
  RegStatesCopy &= Candidates;

  // Returns the first unused (bit is set) register, or 0 is none is found.
  int Reg = RegStatesCopy.find_first();
  return (Reg == -1) ? 0 : Reg;
}

unsigned RegScavenger::FindUnusedReg(const TargetRegisterClass *RegClass,
                                     bool ExCalleeSaved) const {
  // Mask off the registers which are not in the TargetRegisterClass.
  BitVector RegStatesCopy(NumPhysRegs, false);
  CreateRegClassMask(RegClass, RegStatesCopy);
  RegStatesCopy &= RegStates;

  // If looking for a non-callee-saved register, mask off all the callee-saved
  // registers.
  if (ExCalleeSaved)
    RegStatesCopy &= ~CalleeSavedRegs;

  // Returns the first unused (bit is set) register, or 0 is none is found.
  int Reg = RegStatesCopy.find_first();
  return (Reg == -1) ? 0 : Reg;
}
