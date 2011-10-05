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
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define GET_REGINFO_TARGET_DESC
#include "PTXGenRegisterInfo.inc"

using namespace llvm;

PTXRegisterInfo::PTXRegisterInfo(PTXTargetMachine &TM,
                                 const TargetInstrInfo &tii)
  // PTX does not have a return address register.
  : PTXGenRegisterInfo(0), TII(tii) {
}

void PTXRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                          int SPAdj,
                                          RegScavenger *RS) const {
  unsigned Index;
  MachineInstr &MI = *II;
  //MachineBasicBlock &MBB = *MI.getParent();
  //DebugLoc dl = MI.getDebugLoc();
  //MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();

  //unsigned Reg = MRI.createVirtualRegister(PTX::RegF32RegisterClass);

  llvm_unreachable("FrameIndex should have been previously eliminated!");

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

  //MachineInstr* MI2 = BuildMI(MBB, II, dl, TII.get(PTX::LOAD_LOCAL_F32))
  //.addReg(Reg, RegState::Define).addImm(FrameIndex);
  //if (MI2->findFirstPredOperandIdx() == -1) {
  //  MI2->addOperand(MachineOperand::CreateReg(PTX::NoRegister, /*IsDef=*/false));
  //  MI2->addOperand(MachineOperand::CreateImm(PTX::PRED_NORMAL));
  //}
  //MI2->dump();

  //MachineOperand ESOp = MachineOperand::CreateES("__local__");

  // This frame index is post stack slot re-use assignments
  //MI.getOperand(Index).ChangeToRegister(Reg, false);
  MI.getOperand(Index).ChangeToImmediate(FrameIndex);
  //MI.getOperand(Index) = ESOp;
}
