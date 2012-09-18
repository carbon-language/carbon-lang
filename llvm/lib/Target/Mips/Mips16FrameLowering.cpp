//===-- Mips16FrameLowering.cpp - Mips16 Frame Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips16 implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "Mips16FrameLowering.h"
#include "MipsInstrInfo.h"
#include "MCTargetDesc/MipsBaseInfo.h"
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

void Mips16FrameLowering::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const MipsInstrInfo &TII =
    *static_cast<const MipsInstrInfo*>(MF.getTarget().getInstrInfo());
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  uint64_t StackSize = MFI->getStackSize();

  // No need to allocate space on the stack.
  if (StackSize == 0 && !MFI->adjustsStack()) return;

  // Adjust stack.
  if (isInt<16>(-StackSize))
    BuildMI(MBB, MBBI, dl, TII.get(Mips::SaveRaF16)).addImm(StackSize);
}

void Mips16FrameLowering::emitEpilogue(MachineFunction &MF,
                                 MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const MipsInstrInfo &TII =
    *static_cast<const MipsInstrInfo*>(MF.getTarget().getInstrInfo());
  DebugLoc dl = MBBI->getDebugLoc();
  uint64_t StackSize = MFI->getStackSize();

  if (!StackSize)
    return;

  // Adjust stack.
  if (isInt<16>(StackSize))
    // assumes stacksize multiple of 8
    BuildMI(MBB, MBBI, dl, TII.get(Mips::RestoreRaF16)).addImm(StackSize);
}

bool Mips16FrameLowering::
spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MI,
                          const std::vector<CalleeSavedInfo> &CSI,
                          const TargetRegisterInfo *TRI) const {
  // FIXME: implement.
  return true;
}

bool
Mips16FrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  // FIXME: implement.
  return true;
}

void Mips16FrameLowering::
processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                     RegScavenger *RS) const {
}

const MipsFrameLowering *
llvm::createMips16FrameLowering(const MipsSubtarget &ST) {
  return new Mips16FrameLowering(ST);
}
