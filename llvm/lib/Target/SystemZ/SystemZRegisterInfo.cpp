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
#include "llvm/Target/TargetFrameInfo.h"
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
  if (hasFP(MF))
    Reserved.set(SystemZ::R11D);
  Reserved.set(SystemZ::R14D);
  Reserved.set(SystemZ::R15D);
  return Reserved;
}

/// needsFP - Return true if the specified function should have a dedicated
/// frame pointer register.  This is true if the function has variable sized
/// allocas or if frame pointer elimination is disabled.
bool SystemZRegisterInfo::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return DisableFramePointerElim(MF) || MFI->hasVarSizedObjects();
}

void SystemZRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  MBB.erase(I);
}

int SystemZRegisterInfo::getFrameIndexOffset(const MachineFunction &MF,
                                             int FI) const {
  const TargetFrameInfo &TFI = *MF.getTarget().getFrameInfo();
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const SystemZMachineFunctionInfo *SystemZMFI =
    MF.getInfo<SystemZMachineFunctionInfo>();
  int Offset = MFI->getObjectOffset(FI) + MFI->getOffsetAdjustment();
  uint64_t StackSize = MFI->getStackSize();

  // Fixed objects are really located in the "previous" frame.
  if (FI < 0)
    StackSize -= SystemZMFI->getCalleeSavedFrameSize();

  Offset += StackSize - TFI.getOffsetOfLocalArea();

  // Skip the register save area if we generated the stack frame.
  if (StackSize || MFI->hasCalls())
    Offset -= TFI.getOffsetOfLocalArea();

  return Offset;
}

void
SystemZRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
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

  // Offset is a either 12-bit unsigned or 20-bit signed integer.
  // FIXME: handle "too long" displacements.
  int Offset =
    getFrameIndexOffset(MF, FrameIndex) + MI.getOperand(i+1).getImm();

  // Check whether displacement is too long to fit into 12 bit zext field.
  MI.setDesc(TII.getMemoryInstr(MI.getOpcode(), Offset));

  MI.getOperand(i+1).ChangeToImmediate(Offset);
}

void
SystemZRegisterInfo::processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                                       RegScavenger *RS) const {
  // Determine whether R15/R14 will ever be clobbered inside the function. And
  // if yes - mark it as 'callee' saved.
  MachineFrameInfo *FFI = MF.getFrameInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Check whether high FPRs are ever used, if yes - we need to save R15 as
  // well.
  static const unsigned HighFPRs[] = {
    SystemZ::F8L,  SystemZ::F9L,  SystemZ::F10L, SystemZ::F11L,
    SystemZ::F12L, SystemZ::F13L, SystemZ::F14L, SystemZ::F15L,
    SystemZ::F8S,  SystemZ::F9S,  SystemZ::F10S, SystemZ::F11S,
    SystemZ::F12S, SystemZ::F13S, SystemZ::F14S, SystemZ::F15S,
  };

  bool HighFPRsUsed = false;
  for (unsigned i = 0, e = array_lengthof(HighFPRs); i != e; ++i)
    HighFPRsUsed |= MRI.isPhysRegUsed(HighFPRs[i]);

  if (FFI->hasCalls())
    /* FIXME: function is varargs */
    /* FIXME: function grabs RA */
    /* FIXME: function calls eh_return */
    MRI.setPhysRegUsed(SystemZ::R14D);

  if (HighFPRsUsed ||
      FFI->hasCalls() ||
      FFI->getObjectIndexEnd() != 0 || // Contains automatic variables
      FFI->hasVarSizedObjects() // Function calls dynamic alloca's
      /* FIXME: function is varargs */)
    MRI.setPhysRegUsed(SystemZ::R15D);
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

#include "SystemZGenRegisterInfo.inc"
