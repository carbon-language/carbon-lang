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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

/// setUsed - Set the register and its sub-registers as being used.
void RegScavenger::setUsed(unsigned Reg) {
  RegsAvailable.reset(Reg);

  for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
       unsigned SubReg = *SubRegs; ++SubRegs)
    RegsAvailable.reset(SubReg);
}

bool RegScavenger::isAliasUsed(unsigned Reg) const {
  if (isUsed(Reg))
    return true;
  for (const unsigned *R = TRI->getAliasSet(Reg); *R; ++R)
    if (isUsed(*R))
      return true;
  return false;
}

void RegScavenger::initRegState() {
  ScavengedReg = 0;
  ScavengedRC = NULL;
  ScavengeRestore = NULL;

  // All registers started out unused.
  RegsAvailable.set();

  // Reserved registers are always used.
  RegsAvailable ^= ReservedRegs;

  if (!MBB)
    return;

  // Live-in registers are in use.
  for (MachineBasicBlock::const_livein_iterator I = MBB->livein_begin(),
         E = MBB->livein_end(); I != E; ++I)
    setUsed(*I);

  // Pristine CSRs are also unavailable.
  BitVector PR = MBB->getParent()->getFrameInfo()->getPristineRegs(MBB);
  for (int I = PR.find_first(); I>0; I = PR.find_next(I))
    setUsed(I);
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
      assert((!EarlyClobberRegs.test(Reg) || MI->isRegTiedToDefOperand(i)) &&
             "Using an early clobbered register!");
    } else {
      assert(MO.isDef());
#if 0
      // FIXME: Enable this once we've figured out how to correctly transfer
      // implicit kills during codegen passes like the coalescer.
      assert((KillRegs.test(Reg) || isUnused(Reg) ||
              isLiveInButUnusedBefore(Reg, MI, MBB, TRI, MRI)) &&
             "Re-defining a live register!");
#endif
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

unsigned RegScavenger::FindUnusedReg(const TargetRegisterClass *RC) const {
  for (TargetRegisterClass::iterator I = RC->begin(), E = RC->end();
       I != E; ++I)
    if (!isAliasUsed(*I))
      return *I;
  return 0;
}

/// findSurvivorReg - Return the candidate register that is unused for the
/// longest after MBBI. UseMI is set to the instruction where the search
/// stopped.
///
/// No more than InstrLimit instructions are inspected.
///
unsigned RegScavenger::findSurvivorReg(MachineBasicBlock::iterator MI,
                                       BitVector &Candidates,
                                       unsigned InstrLimit,
                                       MachineBasicBlock::iterator &UseMI) {
  int Survivor = Candidates.find_first();
  assert(Survivor > 0 && "No candidates for scavenging");

  MachineBasicBlock::iterator ME = MBB->getFirstTerminator();
  assert(MI != ME && "MI already at terminator");

  for (++MI; InstrLimit > 0 && MI != ME; ++MI, --InstrLimit) {
    // Remove any candidates touched by instruction.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || MO.isUndef() || !MO.getReg() ||
          TargetRegisterInfo::isVirtualRegister(MO.getReg()))
        continue;
      Candidates.reset(MO.getReg());
      for (const unsigned *R = TRI->getAliasSet(MO.getReg()); *R; R++)
        Candidates.reset(*R);
    }

    // Was our survivor untouched by this instruction?
    if (Candidates.test(Survivor))
      continue;

    // All candidates gone?
    if (Candidates.none())
      break;

    Survivor = Candidates.find_first();
  }

  // We ran out of candidates, so stop the search.
  UseMI = MI;
  return Survivor;
}

unsigned RegScavenger::scavengeRegister(const TargetRegisterClass *RC,
                                        MachineBasicBlock::iterator I,
                                        int SPAdj) {
  // Mask off the registers which are not in the TargetRegisterClass.
  BitVector Candidates(NumPhysRegs, false);
  CreateRegClassMask(RC, Candidates);
  // Do not include reserved registers.
  Candidates ^= ReservedRegs & Candidates;

  // Exclude all the registers being used by the instruction.
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = I->getOperand(i);
    if (MO.isReg() && MO.getReg() != 0 &&
        !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
      Candidates.reset(MO.getReg());
  }

  // Find the register whose use is furthest away.
  MachineBasicBlock::iterator UseMI;
  unsigned SReg = findSurvivorReg(I, Candidates, 25, UseMI);

  // If we found an unused register there is no reason to spill it. We have
  // probably found a callee-saved register that has been saved in the
  // prologue, but happens to be unused at this point.
  if (!isAliasUsed(SReg))
    return SReg;

  assert(ScavengedReg == 0 &&
         "Scavenger slot is live, unable to scavenge another register!");

  // Avoid infinite regress
  ScavengedReg = SReg;

  // If the target knows how to save/restore the register, let it do so;
  // otherwise, use the emergency stack spill slot.
  if (!TRI->saveScavengerRegister(*MBB, I, UseMI, RC, SReg)) {
    // Spill the scavenged register before I.
    assert(ScavengingFrameIndex >= 0 &&
           "Cannot scavenge register without an emergency spill slot!");
    TII->storeRegToStackSlot(*MBB, I, SReg, true, ScavengingFrameIndex, RC);
    MachineBasicBlock::iterator II = prior(I);
    TRI->eliminateFrameIndex(II, SPAdj, NULL, this);

    // Restore the scavenged register before its use (or first terminator).
    TII->loadRegFromStackSlot(*MBB, UseMI, SReg, ScavengingFrameIndex, RC);
    II = prior(UseMI);
    TRI->eliminateFrameIndex(II, SPAdj, NULL, this);
  }

  ScavengeRestore = prior(UseMI);

  // Doing this here leads to infinite regress.
  // ScavengedReg = SReg;
  ScavengedRC = RC;

  return SReg;
}
