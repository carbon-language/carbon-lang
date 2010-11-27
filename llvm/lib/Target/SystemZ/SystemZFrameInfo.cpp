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

SystemZFrameInfo::SystemZFrameInfo(const SystemZSubtarget &sti)
  : TargetFrameInfo(TargetFrameInfo::StackGrowsDown, 8, -160), STI(sti) {
  // Fill the spill offsets map
  static const unsigned SpillOffsTab[][2] = {
    { SystemZ::R2D,  0x10 },
    { SystemZ::R3D,  0x18 },
    { SystemZ::R4D,  0x20 },
    { SystemZ::R5D,  0x28 },
    { SystemZ::R6D,  0x30 },
    { SystemZ::R7D,  0x38 },
    { SystemZ::R8D,  0x40 },
    { SystemZ::R9D,  0x48 },
    { SystemZ::R10D, 0x50 },
    { SystemZ::R11D, 0x58 },
    { SystemZ::R12D, 0x60 },
    { SystemZ::R13D, 0x68 },
    { SystemZ::R14D, 0x70 },
    { SystemZ::R15D, 0x78 }
  };

  RegSpillOffsets.grow(SystemZ::NUM_TARGET_REGS);

  for (unsigned i = 0, e = array_lengthof(SpillOffsTab); i != e; ++i)
    RegSpillOffsets[SpillOffsTab[i][0]] = SpillOffsTab[i][1];
}

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

int SystemZFrameInfo::getFrameIndexOffset(const MachineFunction &MF,
                                          int FI) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const SystemZMachineFunctionInfo *SystemZMFI =
    MF.getInfo<SystemZMachineFunctionInfo>();
  int Offset = MFI->getObjectOffset(FI) + MFI->getOffsetAdjustment();
  uint64_t StackSize = MFI->getStackSize();

  // Fixed objects are really located in the "previous" frame.
  if (FI < 0)
    StackSize -= SystemZMFI->getCalleeSavedFrameSize();

  Offset += StackSize - getOffsetOfLocalArea();

  // Skip the register save area if we generated the stack frame.
  if (StackSize || MFI->hasCalls())
    Offset -= getOffsetOfLocalArea();

  return Offset;
}

bool
SystemZFrameInfo::spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                            MachineBasicBlock::iterator MI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                          const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL;
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();
  SystemZMachineFunctionInfo *MFI = MF.getInfo<SystemZMachineFunctionInfo>();
  unsigned CalleeFrameSize = 0;

  // Scan the callee-saved and find the bounds of register spill area.
  unsigned LowReg = 0, HighReg = 0, StartOffset = -1U, EndOffset = 0;
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    if (!SystemZ::FP64RegClass.contains(Reg)) {
      unsigned Offset = RegSpillOffsets[Reg];
      CalleeFrameSize += 8;
      if (StartOffset > Offset) {
        LowReg = Reg; StartOffset = Offset;
      }
      if (EndOffset < Offset) {
        HighReg = Reg; EndOffset = RegSpillOffsets[Reg];
      }
    }
  }

  // Save information for epilogue inserter.
  MFI->setCalleeSavedFrameSize(CalleeFrameSize);
  MFI->setLowReg(LowReg); MFI->setHighReg(HighReg);

  // Save GPRs
  if (StartOffset) {
    // Build a store instruction. Use STORE MULTIPLE instruction if there are many
    // registers to store, otherwise - just STORE.
    MachineInstrBuilder MIB =
      BuildMI(MBB, MI, DL, TII.get((LowReg == HighReg ?
                                    SystemZ::MOV64mr : SystemZ::MOV64mrm)));

    // Add store operands.
    MIB.addReg(SystemZ::R15D).addImm(StartOffset);
    if (LowReg == HighReg)
      MIB.addReg(0);
    MIB.addReg(LowReg, RegState::Kill);
    if (LowReg != HighReg)
      MIB.addReg(HighReg, RegState::Kill);

    // Do a second scan adding regs as being killed by instruction
    for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
      unsigned Reg = CSI[i].getReg();
      // Add the callee-saved register as live-in. It's killed at the spill.
      MBB.addLiveIn(Reg);
      if (Reg != LowReg && Reg != HighReg)
        MIB.addReg(Reg, RegState::ImplicitKill);
    }
  }

  // Save FPRs
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    if (SystemZ::FP64RegClass.contains(Reg)) {
      MBB.addLiveIn(Reg);
      TII.storeRegToStackSlot(MBB, MI, Reg, true, CSI[i].getFrameIdx(),
                              &SystemZ::FP64RegClass, TRI);
    }
  }

  return true;
}

bool
SystemZFrameInfo::restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                              MachineBasicBlock::iterator MI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                          const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL;
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();
  SystemZMachineFunctionInfo *MFI = MF.getInfo<SystemZMachineFunctionInfo>();

  // Restore FP registers
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    if (SystemZ::FP64RegClass.contains(Reg))
      TII.loadRegFromStackSlot(MBB, MI, Reg, CSI[i].getFrameIdx(),
                               &SystemZ::FP64RegClass, TRI);
  }

  // Restore GP registers
  unsigned LowReg = MFI->getLowReg(), HighReg = MFI->getHighReg();
  unsigned StartOffset = RegSpillOffsets[LowReg];

  if (StartOffset) {
    // Build a load instruction. Use LOAD MULTIPLE instruction if there are many
    // registers to load, otherwise - just LOAD.
    MachineInstrBuilder MIB =
      BuildMI(MBB, MI, DL, TII.get((LowReg == HighReg ?
                                    SystemZ::MOV64rm : SystemZ::MOV64rmm)));
    // Add store operands.
    MIB.addReg(LowReg, RegState::Define);
    if (LowReg != HighReg)
      MIB.addReg(HighReg, RegState::Define);

    MIB.addReg(hasFP(MF) ? SystemZ::R11D : SystemZ::R15D);
    MIB.addImm(StartOffset);
    if (LowReg == HighReg)
      MIB.addReg(0);

    // Do a second scan adding regs as being defined by instruction
    for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
      unsigned Reg = CSI[i].getReg();
      if (Reg != LowReg && Reg != HighReg)
        MIB.addReg(Reg, RegState::ImplicitDefine);
    }
  }

  return true;
}

void
SystemZFrameInfo::processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
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
