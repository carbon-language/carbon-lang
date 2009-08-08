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
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
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
    if (!MO.isReg() || MO.isUndef())
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

bool RegScavenger::isSuperRegUsed(unsigned Reg) const {
  for (const unsigned *SuperRegs = TRI->getSuperRegisters(Reg);
       unsigned SuperReg = *SuperRegs; ++SuperRegs)
    if (isUsed(SuperReg))
      return true;
  return false;
}

/// setUsed - Set the register and its sub-registers as being used.
void RegScavenger::setUsed(unsigned Reg) {
  RegsAvailable.reset(Reg);

  for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
       unsigned SubReg = *SubRegs; ++SubRegs)
    RegsAvailable.reset(SubReg);
}

/// setUnused - Set the register and its sub-registers as being unused.
void RegScavenger::setUnused(unsigned Reg, const MachineInstr *MI) {
  RegsAvailable.set(Reg);

  for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
       unsigned SubReg = *SubRegs; ++SubRegs)
    if (!RedefinesSuperRegPart(MI, Reg, TRI))
      RegsAvailable.set(SubReg);
}

void RegScavenger::initRegState() {
  ScavengedReg = 0;
  ScavengedRC = NULL;
  ScavengeRestore = NULL;
  CurrDist = 0;
  DistanceMap.clear();

  // All registers started out unused.
  RegsAvailable.set();

  // Reserved registers are always used.
  RegsAvailable ^= ReservedRegs;

  // Live-in registers are in use.
  if (!MBB || MBB->livein_empty())
    return;
  for (MachineBasicBlock::const_livein_iterator I = MBB->livein_begin(),
         E = MBB->livein_end(); I != E; ++I)
    setUsed(*I);
}

void RegScavenger::enterBasicBlock(MachineBasicBlock *mbb) {
  MachineFunction &MF = *mbb->getParent();
  const TargetMachine &TM = MF.getTarget();
  TII = TM.getInstrInfo();
  TRI = TM.getRegisterInfo();
  MRI = &MF.getRegInfo();

  assert((NumPhysRegs == 0 || NumPhysRegs == TRI->getNumRegs()) &&
         "Target changed?");

  // Self-initialize.
  if (!MBB) {
    NumPhysRegs = TRI->getNumRegs();
    RegsAvailable.resize(NumPhysRegs);

    // Create reserved registers bitvector.
    ReservedRegs = TRI->getReservedRegs(MF);

    // Create callee-saved registers bitvector.
    CalleeSavedRegs.resize(NumPhysRegs);
    const unsigned *CSRegs = TRI->getCalleeSavedRegs();
    if (CSRegs != NULL)
      for (unsigned i = 0; CSRegs[i]; ++i)
        CalleeSavedRegs.set(CSRegs[i]);
  }

  // RS used within emit{Pro,Epi}logue()
  if (mbb != MBB) {
    MBB = mbb;
    initRegState();
  }

  Tracking = false;
}

void RegScavenger::restoreScavengedReg() {
  TII->loadRegFromStackSlot(*MBB, MBBI, ScavengedReg,
                            ScavengingFrameIndex, ScavengedRC);
  MachineBasicBlock::iterator II = prior(MBBI);
  TRI->eliminateFrameIndex(II, 0, this);
  setUsed(ScavengedReg);
  ScavengedReg = 0;
  ScavengedRC = NULL;
}

#ifndef NDEBUG
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
    MachineOperand &UseMO = UI.getOperand();
    if (UseMO.isReg() && UseMO.isUndef())
      continue;
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
#endif

void RegScavenger::addRegWithSubRegs(BitVector &BV, unsigned Reg) {
  BV.set(Reg);
  for (const unsigned *R = TRI->getSubRegisters(Reg); *R; R++)
    BV.set(*R);
}

void RegScavenger::addRegWithAliases(BitVector &BV, unsigned Reg) {
  BV.set(Reg);
  for (const unsigned *R = TRI->getAliasSet(Reg); *R; R++)
    BV.set(*R);
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
  DistanceMap.insert(std::make_pair(MI, CurrDist++));

  if (MI == ScavengeRestore) {
    ScavengedReg = 0;
    ScavengedRC = NULL;
    ScavengeRestore = NULL;
  }

  // Find out which registers are early clobbered, killed, defined, and marked
  // def-dead in this instruction.
  BitVector EarlyClobberRegs(NumPhysRegs);
  BitVector KillRegs(NumPhysRegs);
  BitVector DefRegs(NumPhysRegs);
  BitVector DeadRegs(NumPhysRegs);
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || MO.isUndef())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg || isReserved(Reg))
      continue;

    if (MO.isUse()) {
      // Two-address operands implicitly kill.
      if (MO.isKill() || MI->isRegTiedToDefOperand(i))
        addRegWithSubRegs(KillRegs, Reg);
    } else {
      assert(MO.isDef());
      if (MO.isDead())
        addRegWithSubRegs(DeadRegs, Reg);
      else
        addRegWithSubRegs(DefRegs, Reg);
      if (MO.isEarlyClobber())
        addRegWithAliases(EarlyClobberRegs, Reg);
    }
  }

  // Verify uses and defs.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || MO.isUndef())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg || isReserved(Reg))
      continue;
    if (MO.isUse()) {
      assert(isUsed(Reg) && "Using an undefined register!");
      assert(!EarlyClobberRegs.test(Reg) &&
             "Using an early clobbered register!");
    } else {
      assert(MO.isDef());
      assert((KillRegs.test(Reg) || isUnused(Reg) || isSuperRegUsed(Reg) ||
              isLiveInButUnusedBefore(Reg, MI, MBB, TRI, MRI)) &&
             "Re-defining a live register!");
    }
  }

  // Commit the changes.
  setUnused(KillRegs);
  setUnused(DeadRegs);
  setUsed(DefRegs);
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

/// findFirstUse - Calculate the distance to the first use of the
/// specified register.
MachineInstr*
RegScavenger::findFirstUse(MachineBasicBlock *MBB,
                           MachineBasicBlock::iterator I, unsigned Reg,
                           unsigned &Dist) {
  MachineInstr *UseMI = 0;
  Dist = ~0U;
  for (MachineRegisterInfo::reg_iterator RI = MRI->reg_begin(Reg),
         RE = MRI->reg_end(); RI != RE; ++RI) {
    MachineInstr *UDMI = &*RI;
    if (UDMI->getParent() != MBB)
      continue;
    DenseMap<MachineInstr*, unsigned>::iterator DI = DistanceMap.find(UDMI);
    if (DI == DistanceMap.end()) {
      // If it's not in map, it's below current MI, let's initialize the
      // map.
      I = next(I);
      unsigned Dist = CurrDist + 1;
      while (I != MBB->end()) {
        DistanceMap.insert(std::make_pair(I, Dist++));
        I = next(I);
      }
    }
    DI = DistanceMap.find(UDMI);
    if (DI->second > CurrDist && DI->second < Dist) {
      Dist = DI->second;
      UseMI = UDMI;
    }
  }
  return UseMI;
}

unsigned RegScavenger::scavengeRegister(const TargetRegisterClass *RC,
                                        MachineBasicBlock::iterator I,
                                        int SPAdj) {
  assert(ScavengingFrameIndex >= 0 &&
         "Cannot scavenge a register without an emergency spill slot!");

  // Mask off the registers which are not in the TargetRegisterClass.
  BitVector Candidates(NumPhysRegs, false);
  CreateRegClassMask(RC, Candidates);
  // Do not include reserved registers.
  Candidates ^= ReservedRegs & Candidates;

  // Exclude all the registers being used by the instruction.
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = I->getOperand(i);
    if (MO.isReg())
      Candidates.reset(MO.getReg());
  }

  // Find the register whose use is furthest away.
  unsigned SReg = 0;
  unsigned MaxDist = 0;
  MachineInstr *MaxUseMI = 0;
  int Reg = Candidates.find_first();
  while (Reg != -1) {
    unsigned Dist;
    MachineInstr *UseMI = findFirstUse(MBB, I, Reg, Dist);
    for (const unsigned *AS = TRI->getAliasSet(Reg); *AS; ++AS) {
      unsigned AsDist;
      MachineInstr *AsUseMI = findFirstUse(MBB, I, *AS, AsDist);
      if (AsDist < Dist) {
        Dist = AsDist;
        UseMI = AsUseMI;
      }
    }
    if (Dist >= MaxDist) {
      MaxDist = Dist;
      MaxUseMI = UseMI;
      SReg = Reg;
    }
    Reg = Candidates.find_next(Reg);
  }

  assert(ScavengedReg == 0 &&
         "Scavenger slot is live, unable to scavenge another register!");

  // Avoid infinite regress
  ScavengedReg = SReg;

  // Make sure SReg is marked as used. It could be considered available
  // if it is one of the callee saved registers, but hasn't been spilled.
  if (!isUsed(SReg)) {
    MBB->addLiveIn(SReg);
    setUsed(SReg);
  }

  // Spill the scavenged register before I.
  TII->storeRegToStackSlot(*MBB, I, SReg, true, ScavengingFrameIndex, RC);
  MachineBasicBlock::iterator II = prior(I);
  TRI->eliminateFrameIndex(II, SPAdj, this);

  // Restore the scavenged register before its use (or first terminator).
  II = MaxUseMI
    ? MachineBasicBlock::iterator(MaxUseMI) : MBB->getFirstTerminator();
  TII->loadRegFromStackSlot(*MBB, II, SReg, ScavengingFrameIndex, RC);
  ScavengeRestore = prior(II);
  // Doing this here leads to infinite regress.
  // ScavengedReg = SReg;
  ScavengedRC = RC;

  return SReg;
}
