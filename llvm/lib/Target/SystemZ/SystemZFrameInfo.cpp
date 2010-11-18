//=====- SystemZFrameInfo.cpp - SystemZ Frame Information ------*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SystemZ implementation of TargetFrameInfo class.
//
//===----------------------------------------------------------------------===//

#include "SystemZFrameInfo.h"
#include "SystemZInstrBuilder.h"
#include "SystemZInstrInfo.h"
#include "SystemZMachineFunctionInfo.h"
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


/// needsFP - Return true if the specified function should have a dedicated
/// frame pointer register.  This is true if the function has variable sized
/// allocas or if frame pointer elimination is disabled.
bool SystemZFrameInfo::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return DisableFramePointerElim(MF) || MFI->hasVarSizedObjects();
}

/// emitSPUpdate - Emit a series of instructions to increment / decrement the
/// stack pointer by a constant value.
static
void emitSPUpdate(MachineBasicBlock &MBB, MachineBasicBlock::iterator &MBBI,
                  int64_t NumBytes, const TargetInstrInfo &TII) {
  unsigned Opc; uint64_t Chunk;
  bool isSub = NumBytes < 0;
  uint64_t Offset = isSub ? -NumBytes : NumBytes;

  if (Offset >= (1LL << 15) - 1) {
    Opc = SystemZ::ADD64ri32;
    Chunk = (1LL << 31) - 1;
  } else {
    Opc = SystemZ::ADD64ri16;
    Chunk = (1LL << 15) - 1;
  }

  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  while (Offset) {
    uint64_t ThisVal = (Offset > Chunk) ? Chunk : Offset;
    MachineInstr *MI =
      BuildMI(MBB, MBBI, DL, TII.get(Opc), SystemZ::R15D)
      .addReg(SystemZ::R15D).addImm(isSub ? -ThisVal : ThisVal);
    // The PSW implicit def is dead.
    MI->getOperand(3).setIsDead();
    Offset -= ThisVal;
  }
}

void SystemZFrameInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  const TargetFrameInfo &TFI = *MF.getTarget().getFrameInfo();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const SystemZInstrInfo &TII =
    *static_cast<const SystemZInstrInfo*>(MF.getTarget().getInstrInfo());
  SystemZMachineFunctionInfo *SystemZMFI =
    MF.getInfo<SystemZMachineFunctionInfo>();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  // Get the number of bytes to allocate from the FrameInfo.
  // Note that area for callee-saved stuff is already allocated, thus we need to
  // 'undo' the stack movement.
  uint64_t StackSize = MFI->getStackSize();
  StackSize -= SystemZMFI->getCalleeSavedFrameSize();

  uint64_t NumBytes = StackSize - TFI.getOffsetOfLocalArea();

  // Skip the callee-saved push instructions.
  while (MBBI != MBB.end() &&
         (MBBI->getOpcode() == SystemZ::MOV64mr ||
          MBBI->getOpcode() == SystemZ::MOV64mrm))
    ++MBBI;

  if (MBBI != MBB.end())
    DL = MBBI->getDebugLoc();

  // adjust stack pointer: R15 -= numbytes
  if (StackSize || MFI->hasCalls()) {
    assert(MF.getRegInfo().isPhysRegUsed(SystemZ::R15D) &&
           "Invalid stack frame calculation!");
    emitSPUpdate(MBB, MBBI, -(int64_t)NumBytes, TII);
  }

  if (hasFP(MF)) {
    // Update R11 with the new base value...
    BuildMI(MBB, MBBI, DL, TII.get(SystemZ::MOV64rr), SystemZ::R11D)
      .addReg(SystemZ::R15D);

    // Mark the FramePtr as live-in in every block except the entry.
    for (MachineFunction::iterator I = llvm::next(MF.begin()), E = MF.end();
         I != E; ++I)
      I->addLiveIn(SystemZ::R11D);

  }
}

void SystemZFrameInfo::emitEpilogue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const TargetFrameInfo &TFI = *MF.getTarget().getFrameInfo();
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  const SystemZInstrInfo &TII =
    *static_cast<const SystemZInstrInfo*>(MF.getTarget().getInstrInfo());
  SystemZMachineFunctionInfo *SystemZMFI =
    MF.getInfo<SystemZMachineFunctionInfo>();
  unsigned RetOpcode = MBBI->getOpcode();

  switch (RetOpcode) {
  case SystemZ::RET: break;  // These are ok
  default:
    assert(0 && "Can only insert epilog into returning blocks");
  }

  // Get the number of bytes to allocate from the FrameInfo
  // Note that area for callee-saved stuff is already allocated, thus we need to
  // 'undo' the stack movement.
  uint64_t StackSize =
    MFI->getStackSize() - SystemZMFI->getCalleeSavedFrameSize();
  uint64_t NumBytes = StackSize - TFI.getOffsetOfLocalArea();

  // Skip the final terminator instruction.
  while (MBBI != MBB.begin()) {
    MachineBasicBlock::iterator PI = prior(MBBI);
    --MBBI;
    if (!PI->getDesc().isTerminator())
      break;
  }

  // During callee-saved restores emission stack frame was not yet finialized
  // (and thus - the stack size was unknown). Tune the offset having full stack
  // size in hands.
  if (StackSize || MFI->hasCalls()) {
    assert((MBBI->getOpcode() == SystemZ::MOV64rmm ||
            MBBI->getOpcode() == SystemZ::MOV64rm) &&
           "Expected to see callee-save register restore code");
    assert(MF.getRegInfo().isPhysRegUsed(SystemZ::R15D) &&
           "Invalid stack frame calculation!");

    unsigned i = 0;
    MachineInstr &MI = *MBBI;
    while (!MI.getOperand(i).isImm()) {
      ++i;
      assert(i < MI.getNumOperands() && "Unexpected restore code!");
    }

    uint64_t Offset = NumBytes + MI.getOperand(i).getImm();
    // If Offset does not fit into 20-bit signed displacement field we need to
    // emit some additional code...
    if (Offset > 524287) {
      // Fold the displacement into load instruction as much as possible.
      NumBytes = Offset - 524287;
      Offset = 524287;
      emitSPUpdate(MBB, MBBI, NumBytes, TII);
    }

    MI.getOperand(i).ChangeToImmediate(Offset);
  }
}
