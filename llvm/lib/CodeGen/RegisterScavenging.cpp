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
using namespace llvm;

RegScavenger::RegScavenger(MachineBasicBlock *mbb)
  : MBB(mbb), MBBI(mbb->begin()) {
  const MachineFunction &MF = *MBB->getParent();
  const TargetMachine &TM = MF.getTarget();
  const MRegisterInfo *RegInfo = TM.getRegisterInfo();

  NumPhysRegs = RegInfo->getNumRegs();
  RegStates.resize(NumPhysRegs, true);

  // Create reserved registers bitvector.
  ReservedRegs = RegInfo->getReservedRegs(MF);
  RegStates ^= ReservedRegs;

  // Create callee-saved registers bitvector.
  CalleeSavedRegs.resize(NumPhysRegs);
  const unsigned *CSRegs = RegInfo->getCalleeSavedRegs();
  if (CSRegs != NULL)
    for (unsigned i = 0; CSRegs[i]; ++i)
      CalleeSavedRegs.set(CSRegs[i]);

  // Live-in registers are in use.
  if (!MBB->livein_empty())
    for (MachineBasicBlock::const_livein_iterator I = MBB->livein_begin(),
           E = MBB->livein_end(); I != E; ++I)
      setUsed(*I);
}

void RegScavenger::forward() {
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
    // Skip two-address destination operand.
    if (TID->findTiedToSrcOperand(i) != -1)
      continue;
    unsigned Reg = MO.getReg();
    assert(isUnused(Reg) || isReserved(Reg));
    if (!MO.isDead())
      setUsed(Reg);
  }

  ++MBBI;
}

void RegScavenger::backward() {
  MachineInstr *MI = --MBBI;
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
