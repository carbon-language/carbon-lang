//===- ARMInstrInfo.cpp - ARM Instruction Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARMInstrInfo.h"
#include "ARM.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "ARMGenInstrInfo.inc"
using namespace llvm;

ARMInstrInfo::ARMInstrInfo()
  : TargetInstrInfo(ARMInsts, sizeof(ARMInsts)/sizeof(ARMInsts[0])) {
}

/// Return true if the instruction is a register to register move and
/// leave the source and dest operands in the passed parameters.
///
bool ARMInstrInfo::isMoveInstr(const MachineInstr &MI,
                                 unsigned &SrcReg, unsigned &DstReg) const {
  MachineOpCode oc = MI.getOpcode();
  switch (oc) {
  default:
    return false;
  case ARM::movrr:
    assert(MI.getNumOperands() == 2 &&
	   MI.getOperand(0).isRegister() &&
	   MI.getOperand(1).isRegister() &&
	   "Invalid ARM MOV instruction");
    SrcReg = MI.getOperand(1).getReg();;
    DstReg = MI.getOperand(0).getReg();;
    return true;
  }
}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned ARMInstrInfo::isLoadFromStackSlot(MachineInstr *MI,
                                             int &FrameIndex) const {
  assert(0 && "not implemented");
  return 0;
}

/// isStoreToStackSlot - If the specified machine instruction is a direct
/// store to a stack slot, return the virtual or physical register number of
/// the source reg along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than storing to the stack slot.
unsigned ARMInstrInfo::isStoreToStackSlot(MachineInstr *MI,
                                            int &FrameIndex) const {
  assert(0 && "not implemented");
  return 0;
}
