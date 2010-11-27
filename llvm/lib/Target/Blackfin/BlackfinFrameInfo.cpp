//====- BlackfinFrameInfo.cpp - Blackfin Frame Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Blackfin implementation of TargetFrameInfo class.
//
//===----------------------------------------------------------------------===//

#include "BlackfinFrameInfo.h"
#include "BlackfinInstrInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;


// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
bool BlackfinFrameInfo::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return DisableFramePointerElim(MF) ||
    MFI->adjustsStack() || MFI->hasVarSizedObjects();
}

// Emit a prologue that sets up a stack frame.
// On function entry, R0-R2 and P0 may hold arguments.
// R3, P1, and P2 may be used as scratch registers
void BlackfinFrameInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const BlackfinRegisterInfo *RegInfo =
    static_cast<const BlackfinRegisterInfo*>(MF.getTarget().getRegisterInfo());
  const BlackfinInstrInfo &TII =
    *static_cast<const BlackfinInstrInfo*>(MF.getTarget().getInstrInfo());

  DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  int FrameSize = MFI->getStackSize();
  if (FrameSize%4) {
    FrameSize = (FrameSize+3) & ~3;
    MFI->setStackSize(FrameSize);
  }

  if (!hasFP(MF)) {
    assert(!MFI->adjustsStack() &&
           "FP elimination on a non-leaf function is not supported");
    RegInfo->adjustRegister(MBB, MBBI, dl, BF::SP, BF::P1, -FrameSize);
    return;
  }

  // emit a LINK instruction
  if (FrameSize <= 0x3ffff) {
    BuildMI(MBB, MBBI, dl, TII.get(BF::LINK)).addImm(FrameSize);
    return;
  }

  // Frame is too big, do a manual LINK:
  // [--SP] = RETS;
  // [--SP] = FP;
  // FP = SP;
  // P1 = -FrameSize;
  // SP = SP + P1;
  BuildMI(MBB, MBBI, dl, TII.get(BF::PUSH))
    .addReg(BF::RETS, RegState::Kill);
  BuildMI(MBB, MBBI, dl, TII.get(BF::PUSH))
    .addReg(BF::FP, RegState::Kill);
  BuildMI(MBB, MBBI, dl, TII.get(BF::MOVE), BF::FP)
    .addReg(BF::SP);
  RegInfo->loadConstant(MBB, MBBI, dl, BF::P1, -FrameSize);
  BuildMI(MBB, MBBI, dl, TII.get(BF::ADDpp), BF::SP)
    .addReg(BF::SP, RegState::Kill)
    .addReg(BF::P1, RegState::Kill);

}

void BlackfinFrameInfo::emitEpilogue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const BlackfinRegisterInfo *RegInfo =
    static_cast<const BlackfinRegisterInfo*>(MF.getTarget().getRegisterInfo());
  const BlackfinInstrInfo &TII =
    *static_cast<const BlackfinInstrInfo*>(MF.getTarget().getInstrInfo());
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  DebugLoc dl = MBBI->getDebugLoc();

  int FrameSize = MFI->getStackSize();
  assert(FrameSize%4 == 0 && "Misaligned frame size");

  if (!hasFP(MF)) {
    assert(!MFI->adjustsStack() &&
           "FP elimination on a non-leaf function is not supported");
    RegInfo->adjustRegister(MBB, MBBI, dl, BF::SP, BF::P1, FrameSize);
    return;
  }

  // emit an UNLINK instruction
  BuildMI(MBB, MBBI, dl, TII.get(BF::UNLINK));
}

void BlackfinFrameInfo::
processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                     RegScavenger *RS) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const BlackfinRegisterInfo *RegInfo =
    static_cast<const BlackfinRegisterInfo*>(MF.getTarget().getRegisterInfo());
  const TargetRegisterClass *RC = BF::DPRegisterClass;

  if (RegInfo->requiresRegisterScavenging(MF)) {
    // Reserve a slot close to SP or frame pointer.
    RS->setScavengingFrameIndex(MFI->CreateStackObject(RC->getSize(),
                                                       RC->getAlignment(),
                                                       false));
  }
}
