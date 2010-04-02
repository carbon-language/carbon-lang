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
#include "BlackfinGenInstrInfo.inc"

using namespace llvm;

BlackfinInstrInfo::BlackfinInstrInfo(BlackfinSubtarget &ST)
  : TargetInstrInfoImpl(BlackfinInsts, array_lengthof(BlackfinInsts)),
    RI(ST, *this),
    Subtarget(ST) {}

/// Return true if the instruction is a register to register move and
/// leave the source and dest operands in the passed parameters.
bool BlackfinInstrInfo::isMoveInstr(const MachineInstr &MI,
                                    unsigned &SrcReg,
                                    unsigned &DstReg,
                                    unsigned &SrcSR,
                                    unsigned &DstSR) const {
  SrcSR = DstSR = 0; // No sub-registers.
  switch (MI.getOpcode()) {
  case BF::MOVE:
  case BF::MOVE_ncccc:
  case BF::MOVE_ccncc:
  case BF::MOVECC_zext:
  case BF::MOVECC_nz:
    DstReg = MI.getOperand(0).getReg();
    SrcReg = MI.getOperand(1).getReg();
    return true;
  case BF::SLL16i:
    if (MI.getOperand(2).getImm()!=0)
      return false;
    DstReg = MI.getOperand(0).getReg();
    SrcReg = MI.getOperand(1).getReg();
    return true;
  default:
    return false;
  }
}

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
             const SmallVectorImpl<MachineOperand> &Cond) const {
  // FIXME this should probably have a DebugLoc operand
  DebugLoc DL;

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

static bool inClass(const TargetRegisterClass &Test,
                    unsigned Reg,
                    const TargetRegisterClass *RC) {
  if (TargetRegisterInfo::isPhysicalRegister(Reg))
    return Test.contains(Reg);
  else
    return &Test==RC || Test.hasSubClass(RC);
}

bool BlackfinInstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I,
                                     unsigned DestReg,
                                     unsigned SrcReg,
                                     const TargetRegisterClass *DestRC,
                                     const TargetRegisterClass *SrcRC) const {
  DebugLoc DL;

  if (inClass(BF::ALLRegClass, DestReg, DestRC) &&
      inClass(BF::ALLRegClass, SrcReg,  SrcRC)) {
    BuildMI(MBB, I, DL, get(BF::MOVE), DestReg).addReg(SrcReg);
    return true;
  }

  if (inClass(BF::D16RegClass, DestReg, DestRC) &&
      inClass(BF::D16RegClass, SrcReg,  SrcRC)) {
    BuildMI(MBB, I, DL, get(BF::SLL16i), DestReg).addReg(SrcReg).addImm(0);
    return true;
  }

  if (inClass(BF::AnyCCRegClass, SrcReg, SrcRC) &&
      inClass(BF::DRegClass, DestReg, DestRC)) {
    if (inClass(BF::NotCCRegClass, SrcReg, SrcRC)) {
      BuildMI(MBB, I, DL, get(BF::MOVENCC_z), DestReg).addReg(SrcReg);
      BuildMI(MBB, I, DL, get(BF::BITTGL), DestReg).addReg(DestReg).addImm(0);
    } else {
      BuildMI(MBB, I, DL, get(BF::MOVECC_zext), DestReg).addReg(SrcReg);
    }
    return true;
  }

  if (inClass(BF::AnyCCRegClass, DestReg, DestRC) &&
      inClass(BF::DRegClass, SrcReg,  SrcRC)) {
    if (inClass(BF::NotCCRegClass, DestReg, DestRC))
      BuildMI(MBB, I, DL, get(BF::SETEQri_not), DestReg).addReg(SrcReg);
    else
      BuildMI(MBB, I, DL, get(BF::MOVECC_nz), DestReg).addReg(SrcReg);
    return true;
  }

  if (inClass(BF::NotCCRegClass, DestReg, DestRC) &&
      inClass(BF::JustCCRegClass, SrcReg,  SrcRC)) {
    BuildMI(MBB, I, DL, get(BF::MOVE_ncccc), DestReg).addReg(SrcReg);
    return true;
  }

  if (inClass(BF::JustCCRegClass, DestReg, DestRC) &&
      inClass(BF::NotCCRegClass, SrcReg,  SrcRC)) {
    BuildMI(MBB, I, DL, get(BF::MOVE_ccncc), DestReg).addReg(SrcReg);
    return true;
  }

  llvm_unreachable((std::string("Bad regclasses for reg-to-reg copy: ")+
                    SrcRC->getName() + " -> " + DestRC->getName()).c_str());
  return false;
}

void
BlackfinInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I,
                                       unsigned SrcReg,
                                       bool isKill,
                                       int FI,
                                       const TargetRegisterClass *RC) const {
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
                                        const TargetRegisterClass *RC) const {
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
