//===- SparcInstrInfo.cpp - Sparc Instruction Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Sparc implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "SparcInstrInfo.h"
#include "Sparc.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "SparcGenInstrInfo.inc"
using namespace llvm;

SparcInstrInfo::SparcInstrInfo(SparcSubtarget &ST)
  : TargetInstrInfo(SparcInsts, sizeof(SparcInsts)/sizeof(SparcInsts[0])),
    RI(ST) {
}

static bool isZeroImm(const MachineOperand &op) {
  return op.isImmediate() && op.getImmedValue() == 0;
}

/// Return true if the instruction is a register to register move and
/// leave the source and dest operands in the passed parameters.
///
bool SparcInstrInfo::isMoveInstr(const MachineInstr &MI,
                                 unsigned &SrcReg, unsigned &DstReg) const {
  // We look for 3 kinds of patterns here:
  // or with G0 or 0
  // add with G0 or 0
  // fmovs or FpMOVD (pseudo double move).
  if (MI.getOpcode() == SP::ORrr || MI.getOpcode() == SP::ADDrr) {
    if (MI.getOperand(1).getReg() == SP::G0) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(2).getReg();
      return true;
    } else if (MI.getOperand(2).getReg() == SP::G0) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(1).getReg();
      return true;
    }
  } else if ((MI.getOpcode() == SP::ORri || MI.getOpcode() == SP::ADDri) &&
             isZeroImm(MI.getOperand(2)) && MI.getOperand(1).isRegister()) {
    DstReg = MI.getOperand(0).getReg();
    SrcReg = MI.getOperand(1).getReg();
    return true;
  } else if (MI.getOpcode() == SP::FMOVS || MI.getOpcode() == SP::FpMOVD ||
             MI.getOpcode() == SP::FMOVD) {
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    return true;
  }
  return false;
}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned SparcInstrInfo::isLoadFromStackSlot(MachineInstr *MI,
                                             int &FrameIndex) const {
  if (MI->getOpcode() == SP::LDri ||
      MI->getOpcode() == SP::LDFri ||
      MI->getOpcode() == SP::LDDFri) {
    if (MI->getOperand(1).isFrameIndex() && MI->getOperand(2).isImmediate() &&
        MI->getOperand(2).getImmedValue() == 0) {
      FrameIndex = MI->getOperand(1).getFrameIndex();
      return MI->getOperand(0).getReg();
    }
  }
  return 0;
}

/// isStoreToStackSlot - If the specified machine instruction is a direct
/// store to a stack slot, return the virtual or physical register number of
/// the source reg along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than storing to the stack slot.
unsigned SparcInstrInfo::isStoreToStackSlot(MachineInstr *MI,
                                            int &FrameIndex) const {
  if (MI->getOpcode() == SP::STri ||
      MI->getOpcode() == SP::STFri ||
      MI->getOpcode() == SP::STDFri) {
    if (MI->getOperand(0).isFrameIndex() && MI->getOperand(1).isImmediate() &&
        MI->getOperand(1).getImmedValue() == 0) {
      FrameIndex = MI->getOperand(0).getFrameIndex();
      return MI->getOperand(2).getReg();
    }
  }
  return 0;
}
