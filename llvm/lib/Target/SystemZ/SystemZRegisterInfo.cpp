//===- SystemZRegisterInfo.cpp - SystemZ Register Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SystemZ implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "SystemZ.h"
#include "SystemZInstrInfo.h"
#include "SystemZMachineFunctionInfo.h"
#include "SystemZRegisterInfo.h"
#include "SystemZSubtarget.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/BitVector.h"
using namespace llvm;

SystemZRegisterInfo::SystemZRegisterInfo(SystemZTargetMachine &tm,
                                         const SystemZInstrInfo &tii)
  : SystemZGenRegisterInfo(SystemZ::ADJCALLSTACKUP, SystemZ::ADJCALLSTACKDOWN),
    TM(tm), TII(tii) {
}

const unsigned*
SystemZRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const unsigned CalleeSavedRegs[] = {
    SystemZ::R6D,  SystemZ::R7D,  SystemZ::R8D,  SystemZ::R9D,
    SystemZ::R10D, SystemZ::R11D, SystemZ::R12D, SystemZ::R13D,
    SystemZ::R14D, SystemZ::R15D,
    SystemZ::F8L,  SystemZ::F9L,  SystemZ::F10L, SystemZ::F11L,
    SystemZ::F12L, SystemZ::F13L, SystemZ::F14L, SystemZ::F15L,
    0
  };

  return CalleeSavedRegs;
}

BitVector SystemZRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  if (TFI->hasFP(MF)) {
    // R11D is the frame pointer. Reserve all aliases.
    Reserved.set(SystemZ::R11D);
    Reserved.set(SystemZ::R11W);
    Reserved.set(SystemZ::R10P);
    Reserved.set(SystemZ::R10Q);
  }

  Reserved.set(SystemZ::R14D);
  Reserved.set(SystemZ::R15D);
  Reserved.set(SystemZ::R14W);
  Reserved.set(SystemZ::R15W);
  Reserved.set(SystemZ::R14P);
  Reserved.set(SystemZ::R14Q);
  return Reserved;
}

const TargetRegisterClass*
SystemZRegisterInfo::getMatchingSuperRegClass(const TargetRegisterClass *A,
                                              const TargetRegisterClass *B,
                                              unsigned Idx) const {
  switch(Idx) {
  // Exact sub-classes don't exist for the other sub-register indexes.
  default: return 0;
  case SystemZ::subreg_32bit:
    if (B == SystemZ::ADDR32RegisterClass)
      return A->getSize() == 8 ? SystemZ::ADDR64RegisterClass : 0;
    return A;
  }
}

void SystemZRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  MBB.erase(I);
}

void
SystemZRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                         int SPAdj, RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unxpected");

  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getIndex();

  unsigned BasePtr = (TFI->hasFP(MF) ? SystemZ::R11D : SystemZ::R15D);

  // This must be part of a rri or ri operand memory reference.  Replace the
  // FrameIndex with base register with BasePtr.  Add an offset to the
  // displacement field.
  MI.getOperand(i).ChangeToRegister(BasePtr, false);

  // Offset is a either 12-bit unsigned or 20-bit signed integer.
  // FIXME: handle "too long" displacements.
  int Offset =
    TFI->getFrameIndexOffset(MF, FrameIndex) + MI.getOperand(i+1).getImm();

  // Check whether displacement is too long to fit into 12 bit zext field.
  MI.setDesc(TII.getMemoryInstr(MI.getOpcode(), Offset));

  MI.getOperand(i+1).ChangeToImmediate(Offset);
}

unsigned SystemZRegisterInfo::getRARegister() const {
  assert(0 && "What is the return address register");
  return 0;
}

unsigned
SystemZRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  assert(0 && "What is the frame register");
  return 0;
}

unsigned SystemZRegisterInfo::getEHExceptionRegister() const {
  assert(0 && "What is the exception register");
  return 0;
}

unsigned SystemZRegisterInfo::getEHHandlerRegister() const {
  assert(0 && "What is the exception handler register");
  return 0;
}

int SystemZRegisterInfo::getDwarfRegNum(unsigned RegNum, bool isEH) const {
  assert(0 && "What is the dwarf register number");
  return -1;
}

int SystemZRegisterInfo::getLLVMRegNum(unsigned DwarfRegNo, bool isEH) const {
  assert(0 && "What is the dwarf register number");
  return -1;
}


#include "SystemZGenRegisterInfo.inc"
