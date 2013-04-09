//===-- SparcFrameLowering.cpp - Sparc Frame Information ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Sparc implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "SparcFrameLowering.h"
#include "SparcInstrInfo.h"
#include "SparcMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

void SparcFrameLowering::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const SparcInstrInfo &TII =
    *static_cast<const SparcInstrInfo*>(MF.getTarget().getInstrInfo());
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  // Get the number of bytes to allocate from the FrameInfo
  int NumBytes = (int) MFI->getStackSize();

  if (SubTarget.is64Bit()) {
    // All 64-bit stack frames must be 16-byte aligned, and must reserve space
    // for spilling the 16 window registers at %sp+BIAS..%sp+BIAS+128.
    NumBytes += 128;
    // Frames with calls must also reserve space for 6 outgoing arguments
    // whether they are used or not. LowerCall_64 takes care of that.
    assert(NumBytes % 16 == 0 && "Stack size not 16-byte aligned");
  } else {
    // Emit the correct save instruction based on the number of bytes in
    // the frame. Minimum stack frame size according to V8 ABI is:
    //   16 words for register window spill
    //    1 word for address of returned aggregate-value
    // +  6 words for passing parameters on the stack
    // ----------
    //   23 words * 4 bytes per word = 92 bytes
    NumBytes += 92;

    // Round up to next doubleword boundary -- a double-word boundary
    // is required by the ABI.
    NumBytes = RoundUpToAlignment(NumBytes, 8);
  }
  NumBytes = -NumBytes;

  if (NumBytes >= -4096) {
    BuildMI(MBB, MBBI, dl, TII.get(SP::SAVEri), SP::O6)
      .addReg(SP::O6).addImm(NumBytes);
  } else {
    // Emit this the hard way.  This clobbers G1 which we always know is
    // available here.
    unsigned OffHi = (unsigned)NumBytes >> 10U;
    BuildMI(MBB, MBBI, dl, TII.get(SP::SETHIi), SP::G1).addImm(OffHi);
    // Emit G1 = G1 + I6
    BuildMI(MBB, MBBI, dl, TII.get(SP::ORri), SP::G1)
      .addReg(SP::G1).addImm(NumBytes & ((1 << 10)-1));
    BuildMI(MBB, MBBI, dl, TII.get(SP::SAVErr), SP::O6)
      .addReg(SP::O6).addReg(SP::G1);
  }
}

void SparcFrameLowering::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (!hasReservedCallFrame(MF)) {
    MachineInstr &MI = *I;
    DebugLoc DL = MI.getDebugLoc();
    int Size = MI.getOperand(0).getImm();
    if (MI.getOpcode() == SP::ADJCALLSTACKDOWN)
      Size = -Size;
    const SparcInstrInfo &TII =
      *static_cast<const SparcInstrInfo*>(MF.getTarget().getInstrInfo());
    if (Size)
      BuildMI(MBB, I, DL, TII.get(SP::ADDri), SP::O6).addReg(SP::O6)
        .addImm(Size);
  }
  MBB.erase(I);
}


void SparcFrameLowering::emitEpilogue(MachineFunction &MF,
                                  MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  const SparcInstrInfo &TII =
    *static_cast<const SparcInstrInfo*>(MF.getTarget().getInstrInfo());
  DebugLoc dl = MBBI->getDebugLoc();
  assert(MBBI->getOpcode() == SP::RETL &&
         "Can only put epilog before 'retl' instruction!");
  BuildMI(MBB, MBBI, dl, TII.get(SP::RESTORErr), SP::G0).addReg(SP::G0)
    .addReg(SP::G0);
}
