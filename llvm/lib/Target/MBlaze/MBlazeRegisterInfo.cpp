//===- MBlazeRegisterInfo.cpp - MBlaze Register Information -== -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MBlaze implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mblaze-reg-info"

#include "MBlaze.h"
#include "MBlazeSubtarget.h"
#include "MBlazeRegisterInfo.h"
#include "MBlazeMachineFunction.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm;

MBlazeRegisterInfo::
MBlazeRegisterInfo(const MBlazeSubtarget &ST, const TargetInstrInfo &tii)
  : MBlazeGenRegisterInfo(MBlaze::ADJCALLSTACKDOWN, MBlaze::ADJCALLSTACKUP),
    Subtarget(ST), TII(tii) {}

/// getRegisterNumbering - Given the enum value for some register, e.g.
/// MBlaze::R0, return the number that it corresponds to (e.g. 0).
unsigned MBlazeRegisterInfo::getRegisterNumbering(unsigned RegEnum) {
  switch (RegEnum) {
    case MBlaze::R0  : case MBlaze::F0  : return 0;
    case MBlaze::R1  : case MBlaze::F1  : return 1;
    case MBlaze::R2  : case MBlaze::F2  : return 2;
    case MBlaze::R3  : case MBlaze::F3  : return 3;
    case MBlaze::R4  : case MBlaze::F4  : return 4;
    case MBlaze::R5  : case MBlaze::F5  : return 5;
    case MBlaze::R6  : case MBlaze::F6  : return 6;
    case MBlaze::R7  : case MBlaze::F7  : return 7;
    case MBlaze::R8  : case MBlaze::F8  : return 8;
    case MBlaze::R9  : case MBlaze::F9  : return 9;
    case MBlaze::R10 : case MBlaze::F10 : return 10;
    case MBlaze::R11 : case MBlaze::F11 : return 11;
    case MBlaze::R12 : case MBlaze::F12 : return 12;
    case MBlaze::R13 : case MBlaze::F13 : return 13;
    case MBlaze::R14 : case MBlaze::F14 : return 14;
    case MBlaze::R15 : case MBlaze::F15 : return 15;
    case MBlaze::R16 : case MBlaze::F16 : return 16;
    case MBlaze::R17 : case MBlaze::F17 : return 17;
    case MBlaze::R18 : case MBlaze::F18 : return 18;
    case MBlaze::R19 : case MBlaze::F19 : return 19;
    case MBlaze::R20 : case MBlaze::F20 : return 20;
    case MBlaze::R21 : case MBlaze::F21 : return 21;
    case MBlaze::R22 : case MBlaze::F22 : return 22;
    case MBlaze::R23 : case MBlaze::F23 : return 23;
    case MBlaze::R24 : case MBlaze::F24 : return 24;
    case MBlaze::R25 : case MBlaze::F25 : return 25;
    case MBlaze::R26 : case MBlaze::F26 : return 26;
    case MBlaze::R27 : case MBlaze::F27 : return 27;
    case MBlaze::R28 : case MBlaze::F28 : return 28;
    case MBlaze::R29 : case MBlaze::F29 : return 29;
    case MBlaze::R30 : case MBlaze::F30 : return 30;
    case MBlaze::R31 : case MBlaze::F31 : return 31;
    default: llvm_unreachable("Unknown register number!");
  }
  return 0; // Not reached
}

unsigned MBlazeRegisterInfo::getPICCallReg() {
  return MBlaze::R20;
}

//===----------------------------------------------------------------------===//
// Callee Saved Registers methods
//===----------------------------------------------------------------------===//

/// MBlaze Callee Saved Registers
const unsigned* MBlazeRegisterInfo::
getCalleeSavedRegs(const MachineFunction *MF) const {
  // MBlaze callee-save register range is R20 - R31
  static const unsigned CalleeSavedRegs[] = {
    MBlaze::R20, MBlaze::R21, MBlaze::R22, MBlaze::R23,
    MBlaze::R24, MBlaze::R25, MBlaze::R26, MBlaze::R27,
    MBlaze::R28, MBlaze::R29, MBlaze::R30, MBlaze::R31,
    0
  };

  return CalleeSavedRegs;
}

/// MBlaze Callee Saved Register Classes
const TargetRegisterClass* const* MBlazeRegisterInfo::
getCalleeSavedRegClasses(const MachineFunction *MF) const {
  static const TargetRegisterClass * const CalleeSavedRC[] = {
    &MBlaze::CPURegsRegClass, &MBlaze::CPURegsRegClass,
    &MBlaze::CPURegsRegClass, &MBlaze::CPURegsRegClass,
    &MBlaze::CPURegsRegClass, &MBlaze::CPURegsRegClass,
    &MBlaze::CPURegsRegClass, &MBlaze::CPURegsRegClass,
    &MBlaze::CPURegsRegClass, &MBlaze::CPURegsRegClass,
    &MBlaze::CPURegsRegClass, &MBlaze::CPURegsRegClass,
    0
  };

  return CalleeSavedRC;
}

BitVector MBlazeRegisterInfo::
getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  Reserved.set(MBlaze::R0);
  Reserved.set(MBlaze::R1);
  Reserved.set(MBlaze::R2);
  Reserved.set(MBlaze::R13);
  Reserved.set(MBlaze::R14);
  Reserved.set(MBlaze::R15);
  Reserved.set(MBlaze::R16);
  Reserved.set(MBlaze::R17);
  Reserved.set(MBlaze::R18);
  Reserved.set(MBlaze::R19);
  return Reserved;
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

void MBlazeRegisterInfo::adjustMBlazeStackFrame(MachineFunction &MF) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MBlazeFunctionInfo *MBlazeFI = MF.getInfo<MBlazeFunctionInfo>();

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

  if (hasFP(MF)) {
    MFI->setObjectOffset(MFI->CreateStackObject(RegSize, RegSize, true),
                         StackOffset);
    MBlazeFI->setFPStackOffset(StackOffset);
    TopCPUSavedRegOff = StackOffset;
    StackOffset += RegSize;
  }

  if (MFI->hasCalls()) {
    MFI->setObjectOffset(MFI->CreateStackObject(RegSize, RegSize, true),
                         StackOffset);
    MBlazeFI->setRAStackOffset(StackOffset);
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

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
bool MBlazeRegisterInfo::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return NoFramePointerElim || MFI->hasVarSizedObjects();
}

// This function eliminate ADJCALLSTACKDOWN,
// ADJCALLSTACKUP pseudo instructions
void MBlazeRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  // Simply discard ADJCALLSTACKDOWN, ADJCALLSTACKUP instructions.
  MBB.erase(I);
}

// FrameIndex represent objects inside a abstract stack.
// We must replace FrameIndex with an stack/frame pointer
// direct reference.
unsigned MBlazeRegisterInfo::
eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                    int *Value, RegScavenger *RS) const {
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();

  unsigned i = 0;
  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() &&
           "Instr doesn't have FrameIndex operand!");
  }

  unsigned oi = i == 2 ? 1 : 2;

  DEBUG(errs() << "\nFunction : " << MF.getFunction()->getName() << "\n";
        errs() << "<--------->\n" << MI);

  int FrameIndex = MI.getOperand(i).getIndex();
  int stackSize  = MF.getFrameInfo()->getStackSize();
  int spOffset   = MF.getFrameInfo()->getObjectOffset(FrameIndex);

  DEBUG(errs() << "FrameIndex : " << FrameIndex << "\n"
               << "spOffset   : " << spOffset << "\n"
               << "stackSize  : " << stackSize << "\n");

  // as explained on LowerFormalArguments, detect negative offsets
  // and adjust SPOffsets considering the final stack size.
  int Offset = ((spOffset < 0) ? (stackSize + (-(spOffset+4))) : (spOffset));
  Offset    += MI.getOperand(oi).getImm();

  DEBUG(errs() << "Offset     : " << Offset << "\n" << "<--------->\n");

  MI.getOperand(oi).ChangeToImmediate(Offset);
  MI.getOperand(i).ChangeToRegister(getFrameRegister(MF), false);
  return 0;
}

void MBlazeRegisterInfo::
emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB   = MF.front();
  MachineFrameInfo *MFI    = MF.getFrameInfo();
  MBlazeFunctionInfo *MBlazeFI = MF.getInfo<MBlazeFunctionInfo>();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc dl = (MBBI != MBB.end() ?
                 MBBI->getDebugLoc() : DebugLoc::getUnknownLoc());

  // Get the right frame order for MBlaze.
  adjustMBlazeStackFrame(MF);

  // Get the number of bytes to allocate from the FrameInfo.
  unsigned StackSize = MFI->getStackSize();

  // No need to allocate space on the stack.
  if (StackSize == 0 && !MFI->hasCalls()) return;

  int FPOffset = MBlazeFI->getFPStackOffset();
  int RAOffset = MBlazeFI->getRAStackOffset();

  // Adjust stack : addi R1, R1, -imm
  BuildMI(MBB, MBBI, dl, TII.get(MBlaze::ADDI), MBlaze::R1)
      .addReg(MBlaze::R1).addImm(-StackSize);

  // Save the return address only if the function isnt a leaf one.
  // swi  R15, R1, stack_loc
  if (MFI->hasCalls()) {
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::SWI))
        .addReg(MBlaze::R15).addImm(RAOffset).addReg(MBlaze::R1);
  }

  // if framepointer enabled, save it and set it
  // to point to the stack pointer
  if (hasFP(MF)) {
    // swi  R19, R1, stack_loc
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::SWI))
      .addReg(MBlaze::R19).addImm(FPOffset).addReg(MBlaze::R1);

    // add R19, R1, R0
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::ADD), MBlaze::R19)
      .addReg(MBlaze::R1).addReg(MBlaze::R0);
  }
}

void MBlazeRegisterInfo::
emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  MachineFrameInfo *MFI            = MF.getFrameInfo();
  MBlazeFunctionInfo *MBlazeFI         = MF.getInfo<MBlazeFunctionInfo>();
  DebugLoc dl = MBBI->getDebugLoc();

  // Get the number of bytes from FrameInfo
  int NumBytes = (int) MFI->getStackSize();

  // Get the FI's where RA and FP are saved.
  int FPOffset = MBlazeFI->getFPStackOffset();
  int RAOffset = MBlazeFI->getRAStackOffset();

  // if framepointer enabled, restore it and restore the
  // stack pointer
  if (hasFP(MF)) {
    // add R1, R19, R0
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::ADD), MBlaze::R1)
      .addReg(MBlaze::R19).addReg(MBlaze::R0);

    // lwi  R19, R1, stack_loc
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::LWI), MBlaze::R19)
      .addImm(FPOffset).addReg(MBlaze::R1);
  }

  // Restore the return address only if the function isnt a leaf one.
  // lwi R15, R1, stack_loc
  if (MFI->hasCalls()) {
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::LWI), MBlaze::R15)
      .addImm(RAOffset).addReg(MBlaze::R1);
  }

  // adjust stack.
  // addi R1, R1, imm
  if (NumBytes) {
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::ADDI), MBlaze::R1)
      .addReg(MBlaze::R1).addImm(NumBytes);
  }
}


void MBlazeRegisterInfo::
processFunctionBeforeFrameFinalized(MachineFunction &MF) const {
  // Set the stack offset where GP must be saved/loaded from.
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MBlazeFunctionInfo *MBlazeFI = MF.getInfo<MBlazeFunctionInfo>();
  if (MBlazeFI->needGPSaveRestore())
    MFI->setObjectOffset(MBlazeFI->getGPFI(), MBlazeFI->getGPStackOffset());
}

unsigned MBlazeRegisterInfo::getRARegister() const {
  return MBlaze::R15;
}

unsigned MBlazeRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return hasFP(MF) ? MBlaze::R19 : MBlaze::R1;
}

unsigned MBlazeRegisterInfo::getEHExceptionRegister() const {
  llvm_unreachable("What is the exception register");
  return 0;
}

unsigned MBlazeRegisterInfo::getEHHandlerRegister() const {
  llvm_unreachable("What is the exception handler register");
  return 0;
}

int MBlazeRegisterInfo::getDwarfRegNum(unsigned RegNum, bool isEH) const {
  llvm_unreachable("What is the dwarf register number");
  return -1;
}

#include "MBlazeGenRegisterInfo.inc"

