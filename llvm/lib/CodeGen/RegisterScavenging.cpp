//===-- RegisterScavenging.cpp - Machine register scavenging --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the machine register scavenger. It can provide
// information, such as unused registers, at any point in a machine basic block.
// It also provides a mechanism to make registers available by evicting them to
// spill slots.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reg-scavenging"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

/// RedefinesSuperRegPart - Return true if the specified register is redefining
/// part of a super-register.
static bool RedefinesSuperRegPart(const MachineInstr *MI, unsigned SubReg,
                                  const TargetRegisterInfo *TRI) {
  bool SeenSuperUse = false;
  bool SeenSuperDef = false;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    if (TRI->isSuperRegister(SubReg, MO.getReg())) {
      if (MO.isUse())
        SeenSuperUse = true;
      else if (MO.isImplicit())
        SeenSuperDef = true;
    }
  }

  return SeenSuperDef && SeenSuperUse;
}

static bool RedefinesSuperRegPart(const MachineInstr *MI,
                                  const MachineOperand &MO,
                                  const TargetRegisterInfo *TRI) {
  assert(MO.isReg() && MO.isDef() && "Not a register def!");
  return RedefinesSuperRegPart(MI, MO.getReg(), TRI);
}

/// setUsed - Set the register and its sub-registers as being used.
void RegScavenger::setUsed(unsigned Reg, bool ImpDef) {
  RegsAvailable.reset(Reg);
  ImplicitDefed[Reg] = ImpDef;

  for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
       unsigned SubReg = *SubRegs; ++SubRegs) {
    RegsAvailable.reset(SubReg);
    ImplicitDefed[SubReg] = ImpDef;
  }
}

/// setUnused - Set the register and its sub-registers as being unused.
void RegScavenger::setUnused(unsigned Reg, const MachineInstr *MI) {
  RegsAvailable.set(Reg);
  ImplicitDefed.reset(Reg);

  for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
       unsigned SubReg = *SubRegs; ++SubRegs)
    if (!RedefinesSuperRegPart(MI, Reg, TRI)) {
      RegsAvailable.set(SubReg);
      ImplicitDefed.reset(SubReg);
    }
}

void RegScavenger::enterBasicBlock(MachineBasicBlock *mbb) {
  MachineFunction &MF = *mbb->getParent();
  const TargetMachine &TM = MF.getTarget();
  TII = TM.getInstrInfo();
  TRI = TM.getRegisterInfo();
  MRI = &MF.getRegInfo();

  assert((NumPhysRegs == 0 || NumPhysRegs == TRI->getNumRegs()) &&
         "Target changed?");

  if (!MBB) {
    NumPhysRegs = TRI->getNumRegs();
    RegsAvailable.resize(NumPhysRegs);
    ImplicitDefed.resize(NumPhysRegs);

    // Create reserved registers bitvector.
    ReservedRegs = TRI->getReservedRegs(MF);

    // Create callee-saved registers bitvector.
    CalleeSavedRegs.resize(NumPhysRegs);
    const unsigned *CSRegs = TRI->getCalleeSavedRegs();
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

  TII->loadRegFromStackSlot(*MBB, MBBI, ScavengedReg,
                            ScavengingFrameIndex, ScavengedRC);
  MachineBasicBlock::iterator II = prior(MBBI);
  TRI->eliminateFrameIndex(II, 0, this);
  setUsed(ScavengedReg);
  ScavengedReg = 0;
  ScavengedRC = NULL;
}

/// isLiveInButUnusedBefore - Return true if register is livein the MBB not
/// not used before it reaches the MI that defines register.
static bool isLiveInButUnusedBefore(unsigned Reg, MachineInstr *MI,
                                    MachineBasicBlock *MBB,
                                    const TargetRegisterInfo *TRI,
                                    MachineRegisterInfo* MRI) {
  // First check if register is livein.
  bool isLiveIn = false;
  for (MachineBasicBlock::const_livein_iterator I = MBB->livein_begin(),
         E = MBB->livein_end(); I != E; ++I)
    if (Reg == *I || TRI->isSuperRegister(Reg, *I)) {
      isLiveIn = true;
      break;
    }
  if (!isLiveIn)
    return false;

  // Is there any use of it before the specified MI?
  SmallPtrSet<MachineInstr*, 4> UsesInMBB;
  for (MachineRegisterInfo::use_iterator UI = MRI->use_begin(Reg),
         UE = MRI->use_end(); UI != UE; ++UI) {
    MachineInstr *UseMI = &*UI;
    if (UseMI->getParent() == MBB)
      UsesInMBB.insert(UseMI);
  }
  if (UsesInMBB.empty())
    return true;

  for (MachineBasicBlock::iterator I = MBB->begin(), E = MI; I != E; ++I)
    if (UsesInMBB.count(&*I))
      return false;
  return true;
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
  const TargetInstrDesc &TID = MI->getDesc();

  // Reaching a terminator instruction. Restore a scavenged register (which
  // must be life out.
  if (TID.isTerminator())
    restoreScavengedReg();

  // Process uses first.
  BitVector ChangedRegs(NumPhysRegs);
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isUse())
      continue;

    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;

    if (!isUsed(Reg)) {
      // Register has been scavenged. Restore it!
      if (Reg != ScavengedReg)
        assert(false && "Using an undefined register!");
      else
        restoreScavengedReg();
    }

    if (MO.isKill() && !isReserved(Reg)) {
      ChangedRegs.set(Reg);

      // Mark sub-registers as changed if they aren't defined in the same
      // instruction.
      for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
           unsigned SubReg = *SubRegs; ++SubRegs)
        ChangedRegs.set(SubReg);
    }
  }

  // Change states of all registers after all the uses are processed to guard
  // against multiple uses.
  setUnused(ChangedRegs);

  // Process defs.
  bool IsImpDef = MI->getOpcode() == TargetInstrInfo::IMPLICIT_DEF;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);

    if (!MO.isReg() || !MO.isDef())
      continue;

    unsigned Reg = MO.getReg();

    // If it's dead upon def, then it is now free.
    if (MO.isDead()) {
      setUnused(Reg, MI);
      continue;
    }

    // Skip two-address destination operand.
    if (TID.findTiedToSrcOperand(i) != -1) {
      assert(isUsed(Reg) && "Using an undefined register!");
      continue;
    }

    // Skip is this is merely redefining part of a super-register.
    if (RedefinesSuperRegPart(MI, MO, TRI))
      continue;

    // Implicit def is allowed to "re-define" any register. Similarly,
    // implicitly defined registers can be clobbered.
    assert((isReserved(Reg) || isUnused(Reg) ||
            IsImpDef || isImplicitlyDefined(Reg) ||
            isLiveInButUnusedBefore(Reg, MI, MBB, TRI, MRI)) &&
           "Re-defining a live register!");
    setUsed(Reg, IsImpDef);
  }
}

void RegScavenger::backward() {
  assert(Tracking && "Not tracking states!");
  assert(MBBI != MBB->begin() && "Already at start of basic block!");
  // Move ptr backward.
  MBBI = prior(MBBI);

  MachineInstr *MI = MBBI;
  // Process defs first.
  const TargetInstrDesc &TID = MI->getDesc();
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isDef())
      continue;
    // Skip two-address destination operand.
    if (TID.findTiedToSrcOperand(i) != -1)
      continue;
    unsigned Reg = MO.getReg();
    assert(isUsed(Reg));
    if (!isReserved(Reg))
      setUnused(Reg, MI);
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

    // Set the sub-registers as "used".
    for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
         unsigned SubReg = *SubRegs; ++SubRegs)
      ChangedRegs.set(SubReg);
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
                                  MachineBasicBlock::iterator I, unsigned Reg,
                                  const TargetRegisterInfo *TRI) {
  unsigned Dist = 0;
  I = next(I);
  while (I != MBB->end()) {
    Dist++;
    if (I->readsRegister(Reg, TRI))
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

  // Find the register whose use is furthest away.
  unsigned SReg = 0;
  unsigned MaxDist = 0;
  int Reg = Candidates.find_first();
  while (Reg != -1) {
    unsigned Dist = calcDistanceToUse(MBB, I, Reg, TRI);
    if (Dist >= MaxDist) {
      MaxDist = Dist;
      SReg = Reg;
    }
    Reg = Candidates.find_next(Reg);
  }

  if (ScavengedReg != 0) {
    // First restore previously scavenged register.
    TII->loadRegFromStackSlot(*MBB, I, ScavengedReg,
                              ScavengingFrameIndex, ScavengedRC);
    MachineBasicBlock::iterator II = prior(I);
    TRI->eliminateFrameIndex(II, SPAdj, this);
  }

  TII->storeRegToStackSlot(*MBB, I, SReg, true, ScavengingFrameIndex, RC);
  MachineBasicBlock::iterator II = prior(I);
  TRI->eliminateFrameIndex(II, SPAdj, this);
  ScavengedReg = SReg;
  ScavengedRC = RC;

  return SReg;
}
