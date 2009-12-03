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

const TargetRegisterClass* const*
SystemZRegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  static const TargetRegisterClass * const CalleeSavedRegClasses[] = {
    &SystemZ::GR64RegClass, &SystemZ::GR64RegClass,
    &SystemZ::GR64RegClass, &SystemZ::GR64RegClass,
    &SystemZ::GR64RegClass, &SystemZ::GR64RegClass,
    &SystemZ::GR64RegClass, &SystemZ::GR64RegClass,
    &SystemZ::GR64RegClass, &SystemZ::GR64RegClass,
    &SystemZ::FP64RegClass, &SystemZ::FP64RegClass,
    &SystemZ::FP64RegClass, &SystemZ::FP64RegClass,
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

/// needsFP - Return true if the specified function should have a dedicated
/// frame pointer register.  This is true if the function has variable sized
/// allocas or if frame pointer elimination is disabled.
bool SystemZRegisterInfo::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return NoFramePointerElim || MFI->hasVarSizedObjects();
}

void SystemZRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  MBB.erase(I);
}

int SystemZRegisterInfo::getFrameIndexOffset(MachineFunction &MF, int FI) const {
  const TargetFrameInfo &TFI = *MF.getTarget().getFrameInfo();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  SystemZMachineFunctionInfo *SystemZMFI =
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

unsigned
SystemZRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                         int SPAdj, int *Value,
                                         RegScavenger *RS) const {
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
  int Offset = getFrameIndexOffset(MF, FrameIndex) + MI.getOperand(i+1).getImm();

  // Check whether displacement is too long to fit into 12 bit zext field.
  MI.setDesc(TII.getMemoryInstr(MI.getOpcode(), Offset));

  MI.getOperand(i+1).ChangeToImmediate(Offset);
  return 0;
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

  DebugLoc DL = (MBBI != MBB.end() ? MBBI->getDebugLoc() :
                 DebugLoc::getUnknownLoc());

  while (Offset) {
    uint64_t ThisVal = (Offset > Chunk) ? Chunk : Offset;
    MachineInstr *MI =
      BuildMI(MBB, MBBI, DL, TII.get(Opc), SystemZ::R15D)
      .addReg(SystemZ::R15D).addImm((isSub ? -(int64_t)ThisVal : ThisVal));
    // The PSW implicit def is dead.
    MI->getOperand(3).setIsDead();
    Offset -= ThisVal;
  }
}

void SystemZRegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  const TargetFrameInfo &TFI = *MF.getTarget().getFrameInfo();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  SystemZMachineFunctionInfo *SystemZMFI =
    MF.getInfo<SystemZMachineFunctionInfo>();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc DL = (MBBI != MBB.end() ? MBBI->getDebugLoc() :
                 DebugLoc::getUnknownLoc());

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

void SystemZRegisterInfo::emitEpilogue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const TargetFrameInfo &TFI = *MF.getTarget().getFrameInfo();
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
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
