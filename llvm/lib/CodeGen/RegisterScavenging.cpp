//===-- RegisterScavenging.cpp - Machine register scavenging --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file implements the machine register scavenger. It can provide
/// information, such as unused registers, at any point in a machine basic
/// block. It also provides a mechanism to make registers available by evicting
/// them to spill slots.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

#define DEBUG_TYPE "reg-scavenging"

void RegScavenger::setRegUsed(unsigned Reg, LaneBitmask LaneMask) {
  for (MCRegUnitMaskIterator RUI(Reg, TRI); RUI.isValid(); ++RUI) {
    LaneBitmask UnitMask = (*RUI).second;
    if (UnitMask == 0 || (LaneMask & UnitMask) != 0)
      RegUnitsAvailable.reset((*RUI).first);
  }
}

void RegScavenger::init(MachineBasicBlock &MBB) {
  MachineFunction &MF = *MBB.getParent();
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();

  assert((NumRegUnits == 0 || NumRegUnits == TRI->getNumRegUnits()) &&
         "Target changed?");

  // It is not possible to use the register scavenger after late optimization
  // passes that don't preserve accurate liveness information.
  assert(MRI->tracksLiveness() &&
         "Cannot use register scavenger with inaccurate liveness");

  // Self-initialize.
  if (!this->MBB) {
    NumRegUnits = TRI->getNumRegUnits();
    RegUnitsAvailable.resize(NumRegUnits);
    KillRegUnits.resize(NumRegUnits);
    DefRegUnits.resize(NumRegUnits);
    TmpRegUnits.resize(NumRegUnits);
  }
  this->MBB = &MBB;

  for (SmallVectorImpl<ScavengedInfo>::iterator I = Scavenged.begin(),
         IE = Scavenged.end(); I != IE; ++I) {
    I->Reg = 0;
    I->Restore = nullptr;
  }

  // All register units start out unused.
  RegUnitsAvailable.set();

  // Pristine CSRs are not available.
  BitVector PR = MF.getFrameInfo()->getPristineRegs(MF);
  for (int I = PR.find_first(); I>0; I = PR.find_next(I))
    setRegUsed(I);

  Tracking = false;
}

void RegScavenger::setLiveInsUsed(const MachineBasicBlock &MBB) {
  for (const auto &LI : MBB.liveins())
    setRegUsed(LI.PhysReg, LI.LaneMask);
}

void RegScavenger::enterBasicBlock(MachineBasicBlock &MBB) {
  init(MBB);
  setLiveInsUsed(MBB);
}

void RegScavenger::enterBasicBlockEnd(MachineBasicBlock &MBB) {
  init(MBB);
  // Merge live-ins of successors to get live-outs.
  for (const MachineBasicBlock *Succ : MBB.successors())
    setLiveInsUsed(*Succ);

  // Move internal iterator at the last instruction of the block.
  if (MBB.begin() != MBB.end()) {
    MBBI = std::prev(MBB.end());
    Tracking = true;
  }
}

void RegScavenger::addRegUnits(BitVector &BV, unsigned Reg) {
  for (MCRegUnitIterator RUI(Reg, TRI); RUI.isValid(); ++RUI)
    BV.set(*RUI);
}

void RegScavenger::removeRegUnits(BitVector &BV, unsigned Reg) {
  for (MCRegUnitIterator RUI(Reg, TRI); RUI.isValid(); ++RUI)
    BV.reset(*RUI);
}

void RegScavenger::determineKillsAndDefs() {
  assert(Tracking && "Must be tracking to determine kills and defs");

  MachineInstr &MI = *MBBI;
  assert(!MI.isDebugValue() && "Debug values have no kills or defs");

  // Find out which registers are early clobbered, killed, defined, and marked
  // def-dead in this instruction.
  KillRegUnits.reset();
  DefRegUnits.reset();
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isRegMask()) {
      TmpRegUnits.clear();
      for (unsigned RU = 0, RUEnd = TRI->getNumRegUnits(); RU != RUEnd; ++RU) {
        for (MCRegUnitRootIterator RURI(RU, TRI); RURI.isValid(); ++RURI) {
          if (MO.clobbersPhysReg(*RURI)) {
            TmpRegUnits.set(RU);
            break;
          }
        }
      }

      // Apply the mask.
      KillRegUnits |= TmpRegUnits;
    }
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isPhysicalRegister(Reg) || isReserved(Reg))
      continue;

    if (MO.isUse()) {
      // Ignore undef uses.
      if (MO.isUndef())
        continue;
      if (MO.isKill())
        addRegUnits(KillRegUnits, Reg);
    } else {
      assert(MO.isDef());
      if (MO.isDead())
        addRegUnits(KillRegUnits, Reg);
      else
        addRegUnits(DefRegUnits, Reg);
    }
  }
}

void RegScavenger::unprocess() {
  assert(Tracking && "Cannot unprocess because we're not tracking");

  MachineInstr &MI = *MBBI;
  if (!MI.isDebugValue()) {
    determineKillsAndDefs();

    // Commit the changes.
    setUsed(KillRegUnits);
    setUnused(DefRegUnits);
  }

  if (MBBI == MBB->begin()) {
    MBBI = MachineBasicBlock::iterator(nullptr);
    Tracking = false;
  } else
    --MBBI;
}

void RegScavenger::forward() {
  // Move ptr forward.
  if (!Tracking) {
    MBBI = MBB->begin();
    Tracking = true;
  } else {
    assert(MBBI != MBB->end() && "Already past the end of the basic block!");
    MBBI = std::next(MBBI);
  }
  assert(MBBI != MBB->end() && "Already at the end of the basic block!");

  MachineInstr &MI = *MBBI;

  for (SmallVectorImpl<ScavengedInfo>::iterator I = Scavenged.begin(),
         IE = Scavenged.end(); I != IE; ++I) {
    if (I->Restore != &MI)
      continue;

    I->Reg = 0;
    I->Restore = nullptr;
  }

  if (MI.isDebugValue())
    return;

  determineKillsAndDefs();

  // Verify uses and defs.
#ifndef NDEBUG
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isPhysicalRegister(Reg) || isReserved(Reg))
      continue;
    if (MO.isUse()) {
      if (MO.isUndef())
        continue;
      if (!isRegUsed(Reg)) {
        // Check if it's partial live: e.g.
        // D0 = insert_subreg D0<undef>, S0
        // ... D0
        // The problem is the insert_subreg could be eliminated. The use of
        // D0 is using a partially undef value. This is not *incorrect* since
        // S1 is can be freely clobbered.
        // Ideally we would like a way to model this, but leaving the
        // insert_subreg around causes both correctness and performance issues.
        bool SubUsed = false;
        for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
          if (isRegUsed(*SubRegs)) {
            SubUsed = true;
            break;
          }
        bool SuperUsed = false;
        for (MCSuperRegIterator SR(Reg, TRI); SR.isValid(); ++SR) {
          if (isRegUsed(*SR)) {
            SuperUsed = true;
            break;
          }
        }
        if (!SubUsed && !SuperUsed) {
          MBB->getParent()->verify(nullptr, "In Register Scavenger");
          llvm_unreachable("Using an undefined register!");
        }
        (void)SubUsed;
        (void)SuperUsed;
      }
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
#endif // NDEBUG

  // Commit the changes.
  setUnused(KillRegUnits);
  setUsed(DefRegUnits);
}

void RegScavenger::backward() {
  assert(Tracking && "Must be tracking to determine kills and defs");

  const MachineInstr &MI = *MBBI;
  // Defined or clobbered registers are available now.
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isRegMask()) {
      for (unsigned RU = 0, RUEnd = TRI->getNumRegUnits(); RU != RUEnd;
           ++RU) {
        for (MCRegUnitRootIterator RURI(RU, TRI); RURI.isValid(); ++RURI) {
          if (MO.clobbersPhysReg(*RURI)) {
            RegUnitsAvailable.set(RU);
            break;
          }
        }
      }
    } else if (MO.isReg() && MO.isDef()) {
      unsigned Reg = MO.getReg();
      if (!Reg || TargetRegisterInfo::isVirtualRegister(Reg) ||
          isReserved(Reg))
        continue;
      addRegUnits(RegUnitsAvailable, Reg);
    }
  }
  // Mark read registers as unavailable.
  for (const MachineOperand &MO : MI.uses()) {
    if (MO.isReg() && MO.readsReg()) {
      unsigned Reg = MO.getReg();
      if (!Reg || TargetRegisterInfo::isVirtualRegister(Reg) ||
          isReserved(Reg))
        continue;
      removeRegUnits(RegUnitsAvailable, Reg);
    }
  }

  // Expire scavenge spill frameindex uses.
  for (ScavengedInfo &I : Scavenged) {
    if (I.Restore == &MI) {
      I.Reg = 0;
      I.Restore = nullptr;
    }
  }

  if (MBBI == MBB->begin()) {
    MBBI = MachineBasicBlock::iterator(nullptr);
    Tracking = false;
  } else
    --MBBI;
}

bool RegScavenger::isRegUsed(unsigned Reg, bool includeReserved) const {
  if (includeReserved && isReserved(Reg))
    return true;
  for (MCRegUnitIterator RUI(Reg, TRI); RUI.isValid(); ++RUI)
    if (!RegUnitsAvailable.test(*RUI))
      return true;
  return false;
}

unsigned RegScavenger::FindUnusedReg(const TargetRegisterClass *RC) const {
  for (unsigned Reg : *RC) {
    if (!isRegUsed(Reg)) {
      DEBUG(dbgs() << "Scavenger found unused reg: " << TRI->getName(Reg) <<
            "\n");
      return Reg;
    }
  }
  return 0;
}

BitVector RegScavenger::getRegsAvailable(const TargetRegisterClass *RC) {
  BitVector Mask(TRI->getNumRegs());
  for (unsigned Reg : *RC)
    if (!isRegUsed(Reg))
      Mask.set(Reg);
  return Mask;
}

unsigned RegScavenger::findSurvivorReg(MachineBasicBlock::iterator StartMI,
                                       BitVector &Candidates,
                                       unsigned InstrLimit,
                                       MachineBasicBlock::iterator &UseMI) {
  int Survivor = Candidates.find_first();
  assert(Survivor > 0 && "No candidates for scavenging");

  MachineBasicBlock::iterator ME = MBB->getFirstTerminator();
  assert(StartMI != ME && "MI already at terminator");
  MachineBasicBlock::iterator RestorePointMI = StartMI;
  MachineBasicBlock::iterator MI = StartMI;

  bool inVirtLiveRange = false;
  for (++MI; InstrLimit > 0 && MI != ME; ++MI, --InstrLimit) {
    if (MI->isDebugValue()) {
      ++InstrLimit; // Don't count debug instructions
      continue;
    }
    bool isVirtKillInsn = false;
    bool isVirtDefInsn = false;
    // Remove any candidates touched by instruction.
    for (const MachineOperand &MO : MI->operands()) {
      if (MO.isRegMask())
        Candidates.clearBitsNotInMask(MO.getRegMask());
      if (!MO.isReg() || MO.isUndef() || !MO.getReg())
        continue;
      if (TargetRegisterInfo::isVirtualRegister(MO.getReg())) {
        if (MO.isDef())
          isVirtDefInsn = true;
        else if (MO.isKill())
          isVirtKillInsn = true;
        continue;
      }
      for (MCRegAliasIterator AI(MO.getReg(), TRI, true); AI.isValid(); ++AI)
        Candidates.reset(*AI);
    }
    // If we're not in a virtual reg's live range, this is a valid
    // restore point.
    if (!inVirtLiveRange) RestorePointMI = MI;

    // Update whether we're in the live range of a virtual register
    if (isVirtKillInsn) inVirtLiveRange = false;
    if (isVirtDefInsn) inVirtLiveRange = true;

    // Was our survivor untouched by this instruction?
    if (Candidates.test(Survivor))
      continue;

    // All candidates gone?
    if (Candidates.none())
      break;

    Survivor = Candidates.find_first();
  }
  // If we ran off the end, that's where we want to restore.
  if (MI == ME) RestorePointMI = ME;
  assert(RestorePointMI != StartMI &&
         "No available scavenger restore location!");

  // We ran out of candidates, so stop the search.
  UseMI = RestorePointMI;
  return Survivor;
}

static std::pair<unsigned, MachineBasicBlock::iterator>
findSurvivorBackwards(const TargetRegisterInfo &TRI,
    MachineBasicBlock::iterator From, MachineBasicBlock::iterator To,
    BitVector &Available, BitVector &Candidates) {
  bool FoundTo = false;
  unsigned Survivor = 0;
  MachineBasicBlock::iterator Pos;
  MachineBasicBlock &MBB = *From->getParent();
  MachineBasicBlock::iterator I = From;
  unsigned InstrLimit = 25;
  unsigned InstrCountDown = InstrLimit;
  for (;;) {
    const MachineInstr &MI = *I;
    if (MI.isDebugValue())
      continue;

    // Remove any candidates touched by instruction.
    bool FoundVReg = false;
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isRegMask()) {
        Candidates.clearBitsNotInMask(MO.getRegMask());
        continue;
      }
      if (!MO.isReg() || MO.isUndef())
        continue;
      unsigned Reg = MO.getReg();
      if (TargetRegisterInfo::isVirtualRegister(Reg)) {
        FoundVReg = true;
      } else if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
        for (MCRegAliasIterator AI(Reg, &TRI, true); AI.isValid(); ++AI)
          Candidates.reset(*AI);
      }
    }

    if (I == To) {
      // If one of the available registers survived this long take it.
      Available &= Candidates;
      int Reg = Available.find_first();
      if (Reg != -1)
        return std::make_pair(Reg, MBB.end());
      // Otherwise we will continue up to InstrLimit instructions to fine
      // the register which is not defined/used for the longest time.
      FoundTo = true;
      Pos = To;
    }
    if (FoundTo) {
      if (Survivor == 0 || !Candidates.test(Survivor)) {
        int Reg = Candidates.find_first();
        if (Reg == -1)
          break;
        Survivor = Reg;
      }
      if (--InstrCountDown == 0 || I == MBB.begin())
        break;
      if (FoundVReg) {
        // We found a vreg, reset the InstrLimit counter.
        InstrCountDown = InstrLimit;
        Pos = I;
      }
    }
    --I;
  }

  return std::make_pair(Survivor, Pos);
}

static unsigned getFrameIndexOperandNum(MachineInstr &MI) {
  unsigned i = 0;
  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }
  return i;
}

RegScavenger::ScavengedInfo &
RegScavenger::spill(unsigned Reg, const TargetRegisterClass &RC, int SPAdj,
                    MachineBasicBlock::iterator Before,
                    MachineBasicBlock::iterator &UseMI) {
  // Find an available scavenging slot with size and alignment matching
  // the requirements of the class RC.
  const MachineFunction &MF = *Before->getParent()->getParent();
  const MachineFrameInfo &MFI = *MF.getFrameInfo();
  unsigned NeedSize = RC.getSize();
  unsigned NeedAlign = RC.getAlignment();

  unsigned SI = Scavenged.size(), Diff = UINT_MAX;
  int FIB = MFI.getObjectIndexBegin(), FIE = MFI.getObjectIndexEnd();
  for (unsigned I = 0; I < Scavenged.size(); ++I) {
    if (Scavenged[I].Reg != 0)
      continue;
    // Verify that this slot is valid for this register.
    int FI = Scavenged[I].FrameIndex;
    if (FI < FIB || FI >= FIE)
      continue;
    unsigned S = MFI.getObjectSize(FI);
    unsigned A = MFI.getObjectAlignment(FI);
    if (NeedSize > S || NeedAlign > A)
      continue;
    // Avoid wasting slots with large size and/or large alignment. Pick one
    // that is the best fit for this register class (in street metric).
    // Picking a larger slot than necessary could happen if a slot for a
    // larger register is reserved before a slot for a smaller one. When
    // trying to spill a smaller register, the large slot would be found
    // first, thus making it impossible to spill the larger register later.
    unsigned D = (S-NeedSize) + (A-NeedAlign);
    if (D < Diff) {
      SI = I;
      Diff = D;
    }
  }

  if (SI == Scavenged.size()) {
    // We need to scavenge a register but have no spill slot, the target
    // must know how to do it (if not, we'll assert below).
    Scavenged.push_back(ScavengedInfo(FIE));
  }

  // Avoid infinite regress
  Scavenged[SI].Reg = Reg;

  // If the target knows how to save/restore the register, let it do so;
  // otherwise, use the emergency stack spill slot.
  if (!TRI->saveScavengerRegister(*MBB, Before, UseMI, &RC, Reg)) {
    // Spill the scavenged register before \p Before.
    int FI = Scavenged[SI].FrameIndex;
    if (FI < FIB || FI >= FIE) {
      std::string Msg = std::string("Error while trying to spill ") +
          TRI->getName(Reg) + " from class " + TRI->getRegClassName(&RC) +
          ": Cannot scavenge register without an emergency spill slot!";
      report_fatal_error(Msg.c_str());
    }
    TII->storeRegToStackSlot(*MBB, Before, Reg, true, Scavenged[SI].FrameIndex,
                             &RC, TRI);
    MachineBasicBlock::iterator II = std::prev(Before);

    unsigned FIOperandNum = getFrameIndexOperandNum(*II);
    TRI->eliminateFrameIndex(II, SPAdj, FIOperandNum, this);

    // Restore the scavenged register before its use (or first terminator).
    TII->loadRegFromStackSlot(*MBB, UseMI, Reg, Scavenged[SI].FrameIndex,
                              &RC, TRI);
    II = std::prev(UseMI);

    FIOperandNum = getFrameIndexOperandNum(*II);
    TRI->eliminateFrameIndex(II, SPAdj, FIOperandNum, this);
  }
  return Scavenged[SI];
}

unsigned RegScavenger::scavengeRegister(const TargetRegisterClass *RC,
                                        MachineBasicBlock::iterator I,
                                        int SPAdj) {
  MachineInstr &MI = *I;
  const MachineFunction &MF = *MI.getParent()->getParent();
  // Consider all allocatable registers in the register class initially
  BitVector Candidates = TRI->getAllocatableSet(MF, RC);

  // Exclude all the registers being used by the instruction.
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.getReg() != 0 && !(MO.isUse() && MO.isUndef()) &&
        !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
      Candidates.reset(MO.getReg());
  }

  // Try to find a register that's unused if there is one, as then we won't
  // have to spill.
  BitVector Available = getRegsAvailable(RC);
  Available &= Candidates;
  if (Available.any())
    Candidates = Available;

  // Find the register whose use is furthest away.
  MachineBasicBlock::iterator UseMI;
  unsigned SReg = findSurvivorReg(I, Candidates, 25, UseMI);

  // If we found an unused register there is no reason to spill it.
  if (!isRegUsed(SReg)) {
    DEBUG(dbgs() << "Scavenged register: " << TRI->getName(SReg) << "\n");
    return SReg;
  }

  ScavengedInfo &Scavenged = spill(SReg, *RC, SPAdj, I, UseMI);
  Scavenged.Restore = std::prev(UseMI);

  DEBUG(dbgs() << "Scavenged register (with spill): " << TRI->getName(SReg) <<
        "\n");

  return SReg;
}

unsigned RegScavenger::scavengeRegisterBackwards(const TargetRegisterClass &RC,
                                                 MachineBasicBlock::iterator To,
                                                 bool RestoreAfter, int SPAdj) {
  const MachineBasicBlock &MBB = *To->getParent();
  const MachineFunction &MF = *MBB.getParent();
  // Consider all allocatable registers in the register class initially
  BitVector Candidates = TRI->getAllocatableSet(MF, &RC);

  // Try to find a register that's unused if there is one, as then we won't
  // have to spill.
  BitVector Available = getRegsAvailable(&RC);

  // Find the register whose use is furthest away.
  MachineBasicBlock::iterator UseMI;
  std::pair<unsigned, MachineBasicBlock::iterator> P =
      findSurvivorBackwards(*TRI, MBBI, To, Available, Candidates);
  unsigned Reg = P.first;
  assert(Reg != 0 && "No register left to scavenge!");
  // Found an available register?
  if (!Available.test(Reg)) {
    MachineBasicBlock::iterator ReloadBefore =
      RestoreAfter ? std::next(MBBI) : MBBI;
    DEBUG(dbgs() << "Reload before: " << *ReloadBefore << '\n');
    ScavengedInfo &Scavenged = spill(Reg, RC, SPAdj, P.second, ReloadBefore);
    Scavenged.Restore = std::prev(P.second);
    addRegUnits(RegUnitsAvailable, Reg);
    DEBUG(dbgs() << "Scavenged register with spill: " << PrintReg(Reg, TRI)
          << " until " << *P.second);
  } else {
    DEBUG(dbgs() << "Scavenged free register: " << PrintReg(Reg, TRI) << '\n');
  }
  return Reg;
}
