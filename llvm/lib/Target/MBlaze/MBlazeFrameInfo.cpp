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
#include "InstPrinter/MBlazeInstPrinter.h"
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

namespace llvm {
  cl::opt<bool> DisableStackAdjust(
    "disable-mblaze-stack-adjust",
    cl::init(false),
    cl::desc("Disable MBlaze stack layout adjustment."),
    cl::Hidden);
}

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

static void analyzeFrameIndexes(MachineFunction &MF) {
  if (DisableStackAdjust) return;

  MachineFrameInfo *MFI = MF.getFrameInfo();
  MBlazeFunctionInfo *MBlazeFI = MF.getInfo<MBlazeFunctionInfo>();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  MachineRegisterInfo::livein_iterator LII = MRI.livein_begin();
  MachineRegisterInfo::livein_iterator LIE = MRI.livein_end();
  const SmallVector<int, 16> &LiveInFI = MBlazeFI->getLiveIn();
  SmallVector<MachineInstr*, 16> EraseInstr;

  MachineBasicBlock *MBB = MF.getBlockNumbered(0);
  MachineBasicBlock::iterator MIB = MBB->begin();
  MachineBasicBlock::iterator MIE = MBB->end();

  int StackAdjust = 0;
  int StackOffset = -28;
  for (unsigned i = 0, e = LiveInFI.size(); i < e; ++i) {
    for (MachineBasicBlock::iterator I=MIB; I != MIE; ++I) {
      if (I->getOpcode() != MBlaze::LWI || I->getNumOperands() != 3 ||
          !I->getOperand(1).isFI() || !I->getOperand(0).isReg() ||
          I->getOperand(1).getIndex() != LiveInFI[i]) continue;

      unsigned FIReg = I->getOperand(0).getReg();
      MachineBasicBlock::iterator SI = I;
      for (SI++; SI != MIE; ++SI) {
        if (!SI->getOperand(0).isReg()) continue;
        if (!SI->getOperand(1).isFI()) continue;
        if (SI->getOpcode() != MBlaze::SWI) continue;

        int FI = SI->getOperand(1).getIndex();
        if (SI->getOperand(0).getReg() != FIReg) continue;
        if (MFI->isFixedObjectIndex(FI)) continue;
        if (MFI->getObjectSize(FI) != 4) continue;
        if (SI->getOperand(0).isDef()) break;

        if (SI->getOperand(0).isKill())
          EraseInstr.push_back(I);
        EraseInstr.push_back(SI);
        MBlazeFI->recordLoadArgsFI(FI, StackOffset);
        StackOffset -= 4;
        StackAdjust += 4;
        break;
      }
    }
  }

  for (MachineBasicBlock::iterator I=MBB->begin(), E=MBB->end(); I != E; ++I) {
    if (I->getOpcode() != MBlaze::SWI || I->getNumOperands() != 3 ||
        !I->getOperand(1).isFI() || !I->getOperand(0).isReg() ||
        I->getOperand(1).getIndex() < 0) continue;

    unsigned FIReg = 0;
    for (MachineRegisterInfo::livein_iterator LI = LII; LI != LIE; ++LI) {
      if (I->getOperand(0).getReg() == LI->first) {
        FIReg = LI->first;
        break;
      }
    }

    if (FIReg) {
      int FI = I->getOperand(1).getIndex();
      MBlazeFI->recordLiveIn(FI);

      StackAdjust += 4;
      switch (FIReg) {
      default: llvm_unreachable("invalid incoming parameter!");
      case MBlaze::R5:  MBlazeFI->recordLoadArgsFI(FI, -4); break;
      case MBlaze::R6:  MBlazeFI->recordLoadArgsFI(FI, -8); break;
      case MBlaze::R7:  MBlazeFI->recordLoadArgsFI(FI, -12); break;
      case MBlaze::R8:  MBlazeFI->recordLoadArgsFI(FI, -16); break;
      case MBlaze::R9:  MBlazeFI->recordLoadArgsFI(FI, -20); break;
      case MBlaze::R10: MBlazeFI->recordLoadArgsFI(FI, -24); break;
      }
    }
  }

  for (int i = 0, e = EraseInstr.size(); i < e; ++i)
    MBB->erase(EraseInstr[i]);

  MBlazeFI->setStackAdjust(StackAdjust);
}

static void determineFrameLayout(MachineFunction &MF) {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MBlazeFunctionInfo *MBlazeFI = MF.getInfo<MBlazeFunctionInfo>();

  // Replace the dummy '0' SPOffset by the negative offsets, as explained on
  // LowerFORMAL_ARGUMENTS. Leaving '0' for while is necessary to avoid
  // the approach done by calculateFrameObjectOffsets to the stack frame.
  MBlazeFI->adjustLoadArgsFI(MFI);
  MBlazeFI->adjustStoreVarArgsFI(MFI);

  // Get the number of bytes to allocate from the FrameInfo
  unsigned FrameSize = MFI->getStackSize();
  FrameSize -= MBlazeFI->getStackAdjust();

  // Get the alignments provided by the target, and the maximum alignment
  // (if any) of the fixed frame objects.
  // unsigned MaxAlign = MFI->getMaxAlignment();
  unsigned TargetAlign = MF.getTarget().getFrameInfo()->getStackAlignment();
  unsigned AlignMask = TargetAlign - 1;

  // Make sure the frame is aligned.
  FrameSize = (FrameSize + AlignMask) & ~AlignMask;
  MFI->setStackSize(FrameSize);
}

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
bool MBlazeFrameInfo::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return DisableFramePointerElim(MF) || MFI->hasVarSizedObjects();
}

void MBlazeFrameInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB   = MF.front();
  MachineFrameInfo *MFI    = MF.getFrameInfo();
  const MBlazeInstrInfo &TII =
    *static_cast<const MBlazeInstrInfo*>(MF.getTarget().getInstrInfo());
  MBlazeFunctionInfo *MBlazeFI = MF.getInfo<MBlazeFunctionInfo>();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  // Determine the correct frame layout
  determineFrameLayout(MF);

  // Get the number of bytes to allocate from the FrameInfo.
  unsigned StackSize = MFI->getStackSize();

  // No need to allocate space on the stack.
  if (StackSize == 0 && !MFI->adjustsStack()) return;

  int FPOffset = MBlazeFI->getFPStackOffset();
  int RAOffset = MBlazeFI->getRAStackOffset();

  // Adjust stack : addi R1, R1, -imm
  BuildMI(MBB, MBBI, DL, TII.get(MBlaze::ADDI), MBlaze::R1)
      .addReg(MBlaze::R1).addImm(-StackSize);

  // swi  R15, R1, stack_loc
  if (MFI->adjustsStack()) {
    BuildMI(MBB, MBBI, DL, TII.get(MBlaze::SWI))
        .addReg(MBlaze::R15).addReg(MBlaze::R1).addImm(RAOffset);
  }

  if (hasFP(MF)) {
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
  MBlazeFunctionInfo *MBlazeFI     = MF.getInfo<MBlazeFunctionInfo>();
  const MBlazeInstrInfo &TII =
    *static_cast<const MBlazeInstrInfo*>(MF.getTarget().getInstrInfo());

  DebugLoc dl = MBBI->getDebugLoc();

  // Get the FI's where RA and FP are saved.
  int FPOffset = MBlazeFI->getFPStackOffset();
  int RAOffset = MBlazeFI->getRAStackOffset();

  if (hasFP(MF)) {
    // add R1, R19, R0
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::ADD), MBlaze::R1)
      .addReg(MBlaze::R19).addReg(MBlaze::R0);

    // lwi  R19, R1, stack_loc
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::LWI), MBlaze::R19)
      .addReg(MBlaze::R1).addImm(FPOffset);
  }

  // lwi R15, R1, stack_loc
  if (MFI->adjustsStack()) {
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::LWI), MBlaze::R15)
      .addReg(MBlaze::R1).addImm(RAOffset);
  }

  // Get the number of bytes from FrameInfo
  int StackSize = (int) MFI->getStackSize();

  // addi R1, R1, imm
  if (StackSize) {
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::ADDI), MBlaze::R1)
      .addReg(MBlaze::R1).addImm(StackSize);
  }
}

void MBlazeFrameInfo::processFunctionBeforeCalleeSavedScan(MachineFunction &MF,                                                            RegScavenger *RS)
                                                           const {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MBlazeFunctionInfo *MBlazeFI = MF.getInfo<MBlazeFunctionInfo>();

  if (MFI->adjustsStack()) {
    MBlazeFI->setRAStackOffset(0);
    MFI->CreateFixedObject(4,0,true);
  }

  if (hasFP(MF)) {
    MBlazeFI->setFPStackOffset(4);
    MFI->CreateFixedObject(4,4,true);
  }

  analyzeFrameIndexes(MF);
}
