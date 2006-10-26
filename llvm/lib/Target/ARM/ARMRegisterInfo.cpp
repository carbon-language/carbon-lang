//===- ARMRegisterInfo.cpp - ARM Register Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMRegisterInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Type.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/STLExtras.h"
#include <iostream>
using namespace llvm;

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
//
static bool hasFP(const MachineFunction &MF) {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return NoFramePointerElim || MFI->hasVarSizedObjects();
}

ARMRegisterInfo::ARMRegisterInfo()
  : ARMGenRegisterInfo(ARM::ADJCALLSTACKDOWN, ARM::ADJCALLSTACKUP) {
}

void ARMRegisterInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, int FI,
                    const TargetRegisterClass *RC) const {
  assert (RC == ARM::IntRegsRegisterClass);
  BuildMI(MBB, I, ARM::str, 3).addReg(SrcReg).addImm(0).addFrameIndex(FI);
}

void ARMRegisterInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC) const {
  assert (RC == ARM::IntRegsRegisterClass);
  BuildMI(MBB, I, ARM::ldr, 2, DestReg).addImm(0).addFrameIndex(FI);
}

void ARMRegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I,
                                     unsigned DestReg, unsigned SrcReg,
                                     const TargetRegisterClass *RC) const {
  assert(RC == ARM::IntRegsRegisterClass ||
         RC == ARM::FPRegsRegisterClass  ||
         RC == ARM::DFPRegsRegisterClass);

  if (RC == ARM::IntRegsRegisterClass)
    BuildMI(MBB, I, ARM::MOV, 3, DestReg).addReg(SrcReg).addImm(0)
      .addImm(ARMShift::LSL);
  else if (RC == ARM::FPRegsRegisterClass)
    BuildMI(MBB, I, ARM::FCPYS, 1, DestReg).addReg(SrcReg);
  else
    BuildMI(MBB, I, ARM::FCPYD, 1, DestReg).addReg(SrcReg);
}

MachineInstr *ARMRegisterInfo::foldMemoryOperand(MachineInstr* MI,
                                                   unsigned OpNum,
                                                   int FI) const {
  return NULL;
}

const unsigned* ARMRegisterInfo::getCalleeSaveRegs() const {
  static const unsigned CalleeSaveRegs[] = {
    ARM::R4,  ARM::R5, ARM::R6,  ARM::R7,
    ARM::R8,  ARM::R9, ARM::R10, ARM::R11,
    ARM::R14, 0
  };
  return CalleeSaveRegs;
}

const TargetRegisterClass* const *
ARMRegisterInfo::getCalleeSaveRegClasses() const {
  static const TargetRegisterClass * const CalleeSaveRegClasses[] = {
    &ARM::IntRegsRegClass, &ARM::IntRegsRegClass, &ARM::IntRegsRegClass, &ARM::IntRegsRegClass,
    &ARM::IntRegsRegClass, &ARM::IntRegsRegClass, &ARM::IntRegsRegClass, &ARM::IntRegsRegClass,
    &ARM::IntRegsRegClass, 0
  };
  return CalleeSaveRegClasses;
}

void ARMRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (hasFP(MF)) {
    assert(0);
  }
  MBB.erase(I);
}

void
ARMRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II) const {
  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();

  assert (MI.getOpcode() == ARM::ldr ||
	  MI.getOpcode() == ARM::str ||
	  MI.getOpcode() == ARM::lea_addri);

  unsigned FrameIdx = 2;
  unsigned OffIdx = 1;

  int FrameIndex = MI.getOperand(FrameIdx).getFrameIndex();

  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) +
               MI.getOperand(OffIdx).getImmedValue();

  unsigned StackSize = MF.getFrameInfo()->getStackSize();

  Offset += StackSize;

  assert (Offset >= 0);
  unsigned BaseRegister = hasFP(MF) ? ARM::R11 : ARM::R13;
  if (Offset < 4096) {
    // Replace the FrameIndex with r13
    MI.getOperand(FrameIdx).ChangeToRegister(BaseRegister, false);
    // Replace the ldr offset with Offset
    MI.getOperand(OffIdx).ChangeToImmediate(Offset);
  } else {
    // Insert a set of r12 with the full address
    // r12 = r13 + offset
    MachineBasicBlock *MBB2 = MI.getParent();
    BuildMI(*MBB2, II, ARM::ADD, 4, ARM::R12).addReg(BaseRegister)
      .addImm(Offset).addImm(0).addImm(ARMShift::LSL);

    // Replace the FrameIndex with r12
    MI.getOperand(FrameIdx).ChangeToRegister(ARM::R12, false);
  }
}

void ARMRegisterInfo::
processFunctionBeforeFrameFinalized(MachineFunction &MF) const {}

void ARMRegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo  *MFI = MF.getFrameInfo();
  int           NumBytes = (int) MFI->getStackSize();

  bool HasFP = hasFP(MF);

  if (MFI->hasCalls()) {
    // We reserve argument space for call sites in the function immediately on
    // entry to the current function.  This eliminates the need for add/sub
    // brackets around call sites.
    NumBytes += MFI->getMaxCallFrameSize();
  }

  if (HasFP)
    // Add space for storing the FP
    NumBytes += 4;

  // Align to 8 bytes
  NumBytes = ((NumBytes + 7) / 8) * 8;

  MFI->setStackSize(NumBytes);

  //sub sp, sp, #NumBytes
  BuildMI(MBB, MBBI, ARM::SUB, 4, ARM::R13).addReg(ARM::R13).addImm(NumBytes)
	  .addImm(0).addImm(ARMShift::LSL);

  if (HasFP) {
    BuildMI(MBB, MBBI, ARM::str, 3)
      .addReg(ARM::R11).addImm(0).addReg(ARM::R13);
    BuildMI(MBB, MBBI, ARM::MOV, 3, ARM::R11).addReg(ARM::R13).addImm(0).
      addImm(ARMShift::LSL);
  }
}

void ARMRegisterInfo::emitEpilogue(MachineFunction &MF,
				   MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  assert(MBBI->getOpcode() == ARM::bx &&
         "Can only insert epilog into returning blocks");

  MachineFrameInfo *MFI = MF.getFrameInfo();
  int          NumBytes = (int) MFI->getStackSize();

  if (hasFP(MF)) {
    BuildMI(MBB, MBBI, ARM::MOV, 3, ARM::R13).addReg(ARM::R11).addImm(0).
      addImm(ARMShift::LSL);
    BuildMI(MBB, MBBI, ARM::ldr, 2, ARM::R11).addImm(0).addReg(ARM::R13);
  }

  //add sp, sp, #NumBytes
  BuildMI(MBB, MBBI, ARM::ADD, 4, ARM::R13).addReg(ARM::R13).addImm(NumBytes)
	  .addImm(0).addImm(ARMShift::LSL);
}

unsigned ARMRegisterInfo::getRARegister() const {
  return ARM::R14;
}

unsigned ARMRegisterInfo::getFrameRegister(MachineFunction &MF) const {
  return hasFP(MF) ? ARM::R11 : ARM::R13;
}

#include "ARMGenRegisterInfo.inc"

