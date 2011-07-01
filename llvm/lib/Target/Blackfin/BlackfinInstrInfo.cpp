//===- BlackfinInstrInfo.cpp - Blackfin Instruction Information -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Blackfin implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "BlackfinInstrInfo.h"
#include "BlackfinSubtarget.h"
#include "Blackfin.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_INSTRINFO_CTOR
#define GET_INSTRINFO_MC_DESC
#include "BlackfinGenInstrInfo.inc"

using namespace llvm;

BlackfinInstrInfo::BlackfinInstrInfo(BlackfinSubtarget &ST)
  : BlackfinGenInstrInfo(BF::ADJCALLSTACKDOWN, BF::ADJCALLSTACKUP),
    RI(ST, *this),
    Subtarget(ST) {}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned BlackfinInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                                int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case BF::LOAD32fi:
  case BF::LOAD16fi:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

/// isStoreToStackSlot - If the specified machine instruction is a direct
/// store to a stack slot, return the virtual or physical register number of
/// the source reg along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than storing to the stack slot.
unsigned BlackfinInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                               int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case BF::STORE32fi:
  case BF::STORE16fi:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned BlackfinInstrInfo::
InsertBranch(MachineBasicBlock &MBB,
             MachineBasicBlock *TBB,
             MachineBasicBlock *FBB,
             const SmallVectorImpl<MachineOperand> &Cond,
             DebugLoc DL) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 1 || Cond.size() == 0) &&
         "Branch conditions have one component!");

  if (Cond.empty()) {
    // Unconditional branch?
    assert(!FBB && "Unconditional branch with multiple successors!");
    BuildMI(&MBB, DL, get(BF::JUMPa)).addMBB(TBB);
    return 1;
  }

  // Conditional branch.
  llvm_unreachable("Implement conditional branches!");
}

void BlackfinInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator I, DebugLoc DL,
                                    unsigned DestReg, unsigned SrcReg,
                                    bool KillSrc) const {
  if (BF::ALLRegClass.contains(DestReg, SrcReg)) {
    BuildMI(MBB, I, DL, get(BF::MOVE), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (BF::D16RegClass.contains(DestReg, SrcReg)) {
    BuildMI(MBB, I, DL, get(BF::SLL16i), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc))
      .addImm(0);
    return;
  }

  if (BF::DRegClass.contains(DestReg)) {
    if (SrcReg == BF::NCC) {
      BuildMI(MBB, I, DL, get(BF::MOVENCC_z), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
      BuildMI(MBB, I, DL, get(BF::BITTGL), DestReg).addReg(DestReg).addImm(0);
      return;
    }
    if (SrcReg == BF::CC) {
      BuildMI(MBB, I, DL, get(BF::MOVECC_zext), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
      return;
    }
  }

  if (BF::DRegClass.contains(SrcReg)) {
    if (DestReg == BF::NCC) {
      BuildMI(MBB, I, DL, get(BF::SETEQri_not), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc)).addImm(0);
      return;
    }
    if (DestReg == BF::CC) {
      BuildMI(MBB, I, DL, get(BF::MOVECC_nz), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
      return;
    }
  }


  if (DestReg == BF::NCC && SrcReg == BF::CC) {
    BuildMI(MBB, I, DL, get(BF::MOVE_ncccc), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (DestReg == BF::CC && SrcReg == BF::NCC) {
    BuildMI(MBB, I, DL, get(BF::MOVE_ccncc), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  llvm_unreachable("Bad reg-to-reg copy");
}

static bool inClass(const TargetRegisterClass &Test,
                    unsigned Reg,
                    const TargetRegisterClass *RC) {
  if (TargetRegisterInfo::isPhysicalRegister(Reg))
    return Test.contains(Reg);
  else
    return Test.hasSubClassEq(RC);
}

void
BlackfinInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I,
                                       unsigned SrcReg,
                                       bool isKill,
                                       int FI,
                                       const TargetRegisterClass *RC,
                                       const TargetRegisterInfo *TRI) const {
  DebugLoc DL = I != MBB.end() ? I->getDebugLoc() : DebugLoc();

  if (inClass(BF::DPRegClass, SrcReg, RC)) {
    BuildMI(MBB, I, DL, get(BF::STORE32fi))
      .addReg(SrcReg, getKillRegState(isKill))
      .addFrameIndex(FI)
      .addImm(0);
    return;
  }

  if (inClass(BF::D16RegClass, SrcReg, RC)) {
    BuildMI(MBB, I, DL, get(BF::STORE16fi))
      .addReg(SrcReg, getKillRegState(isKill))
      .addFrameIndex(FI)
      .addImm(0);
    return;
  }

  if (inClass(BF::AnyCCRegClass, SrcReg, RC)) {
    BuildMI(MBB, I, DL, get(BF::STORE8fi))
      .addReg(SrcReg, getKillRegState(isKill))
      .addFrameIndex(FI)
      .addImm(0);
    return;
  }

  llvm_unreachable((std::string("Cannot store regclass to stack slot: ")+
                    RC->getName()).c_str());
}

void BlackfinInstrInfo::
storeRegToAddr(MachineFunction &MF,
               unsigned SrcReg,
               bool isKill,
               SmallVectorImpl<MachineOperand> &Addr,
               const TargetRegisterClass *RC,
               SmallVectorImpl<MachineInstr*> &NewMIs) const {
  llvm_unreachable("storeRegToAddr not implemented");
}

void
BlackfinInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator I,
                                        unsigned DestReg,
                                        int FI,
                                        const TargetRegisterClass *RC,
                                        const TargetRegisterInfo *TRI) const {
  DebugLoc DL = I != MBB.end() ? I->getDebugLoc() : DebugLoc();
  if (inClass(BF::DPRegClass, DestReg, RC)) {
    BuildMI(MBB, I, DL, get(BF::LOAD32fi), DestReg)
      .addFrameIndex(FI)
      .addImm(0);
    return;
  }

  if (inClass(BF::D16RegClass, DestReg, RC)) {
    BuildMI(MBB, I, DL, get(BF::LOAD16fi), DestReg)
      .addFrameIndex(FI)
      .addImm(0);
    return;
  }

  if (inClass(BF::AnyCCRegClass, DestReg, RC)) {
    BuildMI(MBB, I, DL, get(BF::LOAD8fi), DestReg)
      .addFrameIndex(FI)
      .addImm(0);
    return;
  }

  llvm_unreachable("Cannot load regclass from stack slot");
}

void BlackfinInstrInfo::
loadRegFromAddr(MachineFunction &MF,
                unsigned DestReg,
                SmallVectorImpl<MachineOperand> &Addr,
                const TargetRegisterClass *RC,
                SmallVectorImpl<MachineInstr*> &NewMIs) const {
  llvm_unreachable("loadRegFromAddr not implemented");
}
