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
  TII = TM.getInstrInfo();
  RegInfo = TM.getRegisterInfo();

  assert((NumPhysRegs == 0 || NumPhysRegs == RegInfo->getNumRegs()) &&
         "Target changed?");

  if (!MBB) {
    NumPhysRegs = RegInfo->getNumRegs();
    RegsAvailable.resize(NumPhysRegs);

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
  ScavengedReg = 0;
  ScavengedRC = NULL;

  // All registers started out unused.
  RegsAvailable.set();

  // Reserved registers are always used.
  RegsAvailable ^= ReservedRegs;

  // Live-in registers are in use.
  if (!MBB->livein_empty())
    for (MachineBasicBlock::const_livein_iterator I = MBB->livein_begin(),
           E = MBB->livein_end(); I != E; ++I)
      setUsed(*I);

  Tracking = false;
}

void RegScavenger::restoreScavengedReg() {
  if (!ScavengedReg)
    return;

  RegInfo->loadRegFromStackSlot(*MBB, MBBI, ScavengedReg,
                                ScavengingFrameIndex, ScavengedRC);
  MachineBasicBlock::iterator II = prior(MBBI);
  RegInfo->eliminateFrameIndex(II, 0, this);
  setUsed(ScavengedReg);
  ScavengedReg = 0;
  ScavengedRC = NULL;
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

  // Reaching a terminator instruction. Restore a scavenged register (which
  // must be life out.
  if (TII->isTerminatorInstr(MI->getOpcode()))
    restoreScavengedReg();

  // Process uses first.
  BitVector ChangedRegs(NumPhysRegs);
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isUse())
      continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0)
      continue;
    if (!isUsed(Reg)) {
      // Register has been scavenged. Restore it!
      if (Reg != ScavengedReg)
        assert(false);
      else
        restoreScavengedReg();
    }
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
    // If it's dead upon def, then it is now free.
    if (MO.isDead()) {
      setUnused(Reg);
      continue;
    }
    // Skip two-address destination operand.
    if (TID->findTiedToSrcOperand(i) != -1) {
      assert(isUsed(Reg));
      continue;
    }
    assert(isUnused(Reg) || isReserved(Reg));
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

void RegScavenger::getRegsUsed(BitVector &used, bool includeReserved) {
  if (includeReserved)
    used = ~RegsAvailable;
  else
    used = ~RegsAvailable & ~ReservedRegs;
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
  BitVector RegsAvailableCopy(NumPhysRegs, false);
  CreateRegClassMask(RegClass, RegsAvailableCopy);
  RegsAvailableCopy &= RegsAvailable;

  // Restrict the search to candidates.
  RegsAvailableCopy &= Candidates;

  // Returns the first unused (bit is set) register, or 0 is none is found.
  int Reg = RegsAvailableCopy.find_first();
  return (Reg == -1) ? 0 : Reg;
}

unsigned RegScavenger::FindUnusedReg(const TargetRegisterClass *RegClass,
                                     bool ExCalleeSaved) const {
  // Mask off the registers which are not in the TargetRegisterClass.
  BitVector RegsAvailableCopy(NumPhysRegs, false);
  CreateRegClassMask(RegClass, RegsAvailableCopy);
  RegsAvailableCopy &= RegsAvailable;

  // If looking for a non-callee-saved register, mask off all the callee-saved
  // registers.
  if (ExCalleeSaved)
    RegsAvailableCopy &= ~CalleeSavedRegs;

  // Returns the first unused (bit is set) register, or 0 is none is found.
  int Reg = RegsAvailableCopy.find_first();
  return (Reg == -1) ? 0 : Reg;
}

/// calcDistanceToUse - Calculate the distance to the first use of the
/// specified register.
static unsigned calcDistanceToUse(MachineBasicBlock *MBB,
                                  MachineBasicBlock::iterator I, unsigned Reg) {
  unsigned Dist = 0;
  I = next(I);
  while (I != MBB->end()) {
    Dist++;
    if (I->findRegisterUseOperandIdx(Reg) != -1)
        return Dist;
    I = next(I);    
  }
  return Dist + 1;
}

unsigned RegScavenger::scavengeRegister(const TargetRegisterClass *RC,
                                        MachineBasicBlock::iterator I,
                                        int SPAdj) {
  assert(ScavengingFrameIndex >= 0 &&
         "Cannot scavenge a register without an emergency spill slot!");

  // Mask off the registers which are not in the TargetRegisterClass.
  BitVector Candidates(NumPhysRegs, false);
  CreateRegClassMask(RC, Candidates);
  Candidates ^= ReservedRegs;  // Do not include reserved registers.

  // Exclude all the registers being used by the instruction.
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = I->getOperand(i);
    if (MO.isReg())
      Candidates.reset(MO.getReg());
  }

  // Find the register whose use is furtherest aaway.
  unsigned SReg = 0;
  unsigned MaxDist = 0;
  int Reg = Candidates.find_first();
  while (Reg != -1) {
    unsigned Dist = calcDistanceToUse(MBB, I, Reg);
    if (Dist >= MaxDist) {
      MaxDist = Dist;
      SReg = Reg;
    }
    Reg = Candidates.find_next(Reg);
  }

  if (ScavengedReg != 0) {
    // First restore previously scavenged register.
    RegInfo->loadRegFromStackSlot(*MBB, I, ScavengedReg,
                                  ScavengingFrameIndex, ScavengedRC);
    MachineBasicBlock::iterator II = prior(I);
    RegInfo->eliminateFrameIndex(II, SPAdj, this);
  }

  RegInfo->storeRegToStackSlot(*MBB, I, SReg, ScavengingFrameIndex, RC);
  MachineBasicBlock::iterator II = prior(I);
  RegInfo->eliminateFrameIndex(II, SPAdj, this);
  ScavengedReg = SReg;
  ScavengedRC = RC;

  return SReg;
}
