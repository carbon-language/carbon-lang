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
#include "SystemZRegisterInfo.h"
#include "SystemZSubtarget.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/BitVector.h"
using namespace llvm;

SystemZRegisterInfo::SystemZRegisterInfo(SystemZTargetMachine &tm,
                                         const TargetInstrInfo &tii)
  : SystemZGenRegisterInfo(SystemZ::NOP, SystemZ::NOP),
    TM(tm), TII(tii) {
}

const unsigned*
SystemZRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const unsigned CalleeSavedRegs[] = {
    SystemZ::R6D,  SystemZ::R7D,  SystemZ::R8D,  SystemZ::R9D,
    SystemZ::R10D, SystemZ::R11D, SystemZ::R12D, SystemZ::R13D,
    SystemZ::F1,  SystemZ::F3,  SystemZ::F5,  SystemZ::F7,
    0
  };

  return CalleeSavedRegs;
}

const TargetRegisterClass* const*
SystemZRegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  static const TargetRegisterClass * const CalleeSavedRegClasses[] = {
    &SystemZ::GR64RegClass, &SystemZ::GR64RegClass,
    &SystemZ::GR64RegClass, &SystemZ::GR64RegClass,
    &SystemZ::GR64RegClass, &SystemZ::GR64RegClass,
    &SystemZ::GR64RegClass, &SystemZ::GR64RegClass,
    &SystemZ::FP64RegClass, &SystemZ::FP64RegClass,
    &SystemZ::FP64RegClass, &SystemZ::FP64RegClass, 0
  };
  return CalleeSavedRegClasses;
}

BitVector SystemZRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  if (hasFP(MF))
    Reserved.set(SystemZ::R11D);
  Reserved.set(SystemZ::R14D);
  Reserved.set(SystemZ::R15D);
  return Reserved;
}

// needsFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
//
bool SystemZRegisterInfo::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return NoFramePointerElim || MFI->hasVarSizedObjects();
}

void SystemZRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  assert(0 && "Not implemented yet!");
}

int SystemZRegisterInfo::getFrameIndexOffset(MachineFunction &MF, int FI) const {
  const TargetFrameInfo &TFI = *MF.getTarget().getFrameInfo();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  int Offset = MFI->getObjectOffset(FI) + MFI->getOffsetAdjustment();
  uint64_t StackSize = MFI->getStackSize();

  Offset += StackSize - TFI.getOffsetOfLocalArea();

  // Skip the register save area if we generated the stack frame.
  if (StackSize)
    Offset -= TFI.getOffsetOfLocalArea();

  return Offset;
}

void SystemZRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                            int SPAdj, RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unxpected");

  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getIndex();

  unsigned BasePtr = (hasFP(MF) ? SystemZ::R11D : SystemZ::R15D);

  // This must be part of a rri or ri operand memory reference.  Replace the
  // FrameIndex with base register with BasePtr.  Add an offset to the
  // displacement field.
  MI.getOperand(i).ChangeToRegister(BasePtr, false);

  // Offset is a 20-bit integer.
  // FIXME: handle "too long" displacements.
  int Offset = getFrameIndexOffset(MF, FrameIndex) + MI.getOperand(i+1).getImm();
  MI.getOperand(i+1).ChangeToImmediate(Offset);
}

void SystemZRegisterInfo::emitPrologue(MachineFunction &MF) const {
  // Nothing here yet
}

void SystemZRegisterInfo::emitEpilogue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  // Nothing here yet
}

unsigned SystemZRegisterInfo::getRARegister() const {
  assert(0 && "What is the return address register");
  return 0;
}

unsigned SystemZRegisterInfo::getFrameRegister(MachineFunction &MF) const {
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

#include "SystemZGenRegisterInfo.inc"
