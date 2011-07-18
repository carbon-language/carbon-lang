//===- PTXRegisterInfo.cpp - PTX Register Information ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PTX implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "PTX.h"
#include "PTXRegisterInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define GET_REGINFO_TARGET_DESC
#include "PTXGenRegisterInfo.inc"

using namespace llvm;

PTXRegisterInfo::PTXRegisterInfo(PTXTargetMachine &TM,
                                 const TargetInstrInfo &TII)
  // PTX does not have a return address register.
  : PTXGenRegisterInfo(0) {
}

void PTXRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                          int SPAdj,
                                          RegScavenger *RS) const {
  unsigned Index;
  MachineInstr& MI = *II;

  Index = 0;
  while (!MI.getOperand(Index).isFI()) {
    ++Index;
    assert(Index < MI.getNumOperands() &&
           "Instr does not have a FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(Index).getIndex();

  DEBUG(dbgs() << "eliminateFrameIndex: " << MI);
  DEBUG(dbgs() << "- SPAdj: " << SPAdj << "\n");
  DEBUG(dbgs() << "- FrameIndex: " << FrameIndex << "\n");

  // This frame index is post stack slot re-use assignments
  MI.getOperand(Index).ChangeToImmediate(FrameIndex);
}
