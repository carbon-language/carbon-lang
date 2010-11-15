//=======- MBlazeFrameInfo.cpp - MBlaze Frame Information ------*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MBlaze implementation of TargetFrameInfo class.
//
//===----------------------------------------------------------------------===//

#include "MBlazeFrameInfo.h"
#include "MBlazeInstrInfo.h"
#include "MBlazeMachineFunction.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;


//===----------------------------------------------------------------------===//
//
// Stack Frame Processing methods
// +----------------------------+
//
// The stack is allocated decrementing the stack pointer on
// the first instruction of a function prologue. Once decremented,
// all stack references are are done through a positive offset
// from the stack/frame pointer, so the stack is considered
// to grow up.
//
//===----------------------------------------------------------------------===//

void MBlazeFrameInfo::adjustMBlazeStackFrame(MachineFunction &MF) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MBlazeFunctionInfo *MBlazeFI = MF.getInfo<MBlazeFunctionInfo>();
  const MBlazeRegisterInfo *RegInfo =
    static_cast<const MBlazeRegisterInfo*>(MF.getTarget().getRegisterInfo());

  // See the description at MicroBlazeMachineFunction.h
  int TopCPUSavedRegOff = -1;

  // Adjust CPU Callee Saved Registers Area. Registers RA and FP must
  // be saved in this CPU Area there is the need. This whole Area must
  // be aligned to the default Stack Alignment requirements.
  unsigned StackOffset = MFI->getStackSize();
  unsigned RegSize = 4;

  // Replace the dummy '0' SPOffset by the negative offsets, as explained on
  // LowerFORMAL_ARGUMENTS. Leaving '0' for while is necessary to avoid
  // the approach done by calculateFrameObjectOffsets to the stack frame.
  MBlazeFI->adjustLoadArgsFI(MFI);
  MBlazeFI->adjustStoreVarArgsFI(MFI);

  if (RegInfo->hasFP(MF)) {
    MFI->setObjectOffset(MFI->CreateStackObject(RegSize, RegSize, true),
                         StackOffset);
    MBlazeFI->setFPStackOffset(StackOffset);
    TopCPUSavedRegOff = StackOffset;
    StackOffset += RegSize;
  }

  if (MFI->adjustsStack()) {
    MBlazeFI->setRAStackOffset(0);
    MFI->setObjectOffset(MFI->CreateStackObject(RegSize, RegSize, true),
                         StackOffset);
    TopCPUSavedRegOff = StackOffset;
    StackOffset += RegSize;
  }

  // Update frame info
  MFI->setStackSize(StackOffset);

  // Recalculate the final tops offset. The final values must be '0'
  // if there isn't a callee saved register for CPU or FPU, otherwise
  // a negative offset is needed.
  if (TopCPUSavedRegOff >= 0)
    MBlazeFI->setCPUTopSavedRegOff(TopCPUSavedRegOff-StackOffset);
}

void MBlazeFrameInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB   = MF.front();
  MachineFrameInfo *MFI    = MF.getFrameInfo();
  const MBlazeRegisterInfo *RegInfo =
    static_cast<const MBlazeRegisterInfo*>(MF.getTarget().getRegisterInfo());
  const MBlazeInstrInfo &TII =
    *static_cast<const MBlazeInstrInfo*>(MF.getTarget().getInstrInfo());
  MBlazeFunctionInfo *MBlazeFI = MF.getInfo<MBlazeFunctionInfo>();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  // Get the right frame order for MBlaze.
  adjustMBlazeStackFrame(MF);

  // Get the number of bytes to allocate from the FrameInfo.
  unsigned StackSize = MFI->getStackSize();

  // No need to allocate space on the stack.
  if (StackSize == 0 && !MFI->adjustsStack()) return;
  if (StackSize < 28 && MFI->adjustsStack()) StackSize = 28;

  int FPOffset = MBlazeFI->getFPStackOffset();
  int RAOffset = MBlazeFI->getRAStackOffset();

  // Adjust stack : addi R1, R1, -imm
  BuildMI(MBB, MBBI, DL, TII.get(MBlaze::ADDI), MBlaze::R1)
      .addReg(MBlaze::R1).addImm(-StackSize);

  // Save the return address only if the function isnt a leaf one.
  // swi  R15, R1, stack_loc
  if (MFI->adjustsStack()) {
    BuildMI(MBB, MBBI, DL, TII.get(MBlaze::SWI))
        .addReg(MBlaze::R15).addReg(MBlaze::R1).addImm(RAOffset);
  }

  // if framepointer enabled, save it and set it
  // to point to the stack pointer
  if (RegInfo->hasFP(MF)) {
    // swi  R19, R1, stack_loc
    BuildMI(MBB, MBBI, DL, TII.get(MBlaze::SWI))
      .addReg(MBlaze::R19).addReg(MBlaze::R1).addImm(FPOffset);

    // add R19, R1, R0
    BuildMI(MBB, MBBI, DL, TII.get(MBlaze::ADD), MBlaze::R19)
      .addReg(MBlaze::R1).addReg(MBlaze::R0);
  }
}

void MBlazeFrameInfo::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  MachineFrameInfo *MFI            = MF.getFrameInfo();
  MBlazeFunctionInfo *MBlazeFI         = MF.getInfo<MBlazeFunctionInfo>();
  const MBlazeRegisterInfo *RegInfo =
    static_cast<const MBlazeRegisterInfo*>(MF.getTarget().getRegisterInfo());
  const MBlazeInstrInfo &TII =
    *static_cast<const MBlazeInstrInfo*>(MF.getTarget().getInstrInfo());

  DebugLoc dl = MBBI->getDebugLoc();

  // Get the FI's where RA and FP are saved.
  int FPOffset = MBlazeFI->getFPStackOffset();
  int RAOffset = MBlazeFI->getRAStackOffset();

  // if framepointer enabled, restore it and restore the
  // stack pointer
  if (RegInfo->hasFP(MF)) {
    // add R1, R19, R0
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::ADD), MBlaze::R1)
      .addReg(MBlaze::R19).addReg(MBlaze::R0);

    // lwi  R19, R1, stack_loc
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::LWI), MBlaze::R19)
      .addReg(MBlaze::R1).addImm(FPOffset);
  }

  // Restore the return address only if the function isnt a leaf one.
  // lwi R15, R1, stack_loc
  if (MFI->adjustsStack()) {
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::LWI), MBlaze::R15)
      .addReg(MBlaze::R1).addImm(RAOffset);
  }

  // Get the number of bytes from FrameInfo
  int StackSize = (int) MFI->getStackSize();
  if (StackSize < 28 && MFI->adjustsStack()) StackSize = 28;

  // adjust stack.
  // addi R1, R1, imm
  if (StackSize) {
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::ADDI), MBlaze::R1)
      .addReg(MBlaze::R1).addImm(StackSize);
  }
}
