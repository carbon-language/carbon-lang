//===- MipsRegisterInfo.cpp - MIPS Register Information -== -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MIPS implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mips-reg-info"

#include "Mips.h"
#include "MipsSubtarget.h"
#include "MipsRegisterInfo.h"
#include "MipsMachineFunction.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/DebugInfo.h"

#define GET_REGINFO_TARGET_DESC
#include "MipsGenRegisterInfo.inc"

using namespace llvm;

MipsRegisterInfo::MipsRegisterInfo(const MipsSubtarget &ST,
                                   const TargetInstrInfo &tii)
  : MipsGenRegisterInfo(Mips::RA), Subtarget(ST), TII(tii) {}

/// getRegisterNumbering - Given the enum value for some register, e.g.
/// Mips::RA, return the number that it corresponds to (e.g. 31).
unsigned MipsRegisterInfo::
getRegisterNumbering(unsigned RegEnum)
{
  switch (RegEnum) {
    case Mips::ZERO : case Mips::F0 : case Mips::D0 : return 0;
    case Mips::AT   : case Mips::F1 : return 1;
    case Mips::V0   : case Mips::F2 : case Mips::D1 : return 2;
    case Mips::V1   : case Mips::F3 : return 3;
    case Mips::A0   : case Mips::F4 : case Mips::D2 : return 4;
    case Mips::A1   : case Mips::F5 : return 5;
    case Mips::A2   : case Mips::F6 : case Mips::D3 : return 6;
    case Mips::A3   : case Mips::F7 : return 7;
    case Mips::T0   : case Mips::F8 : case Mips::D4 : return 8;
    case Mips::T1   : case Mips::F9 : return 9;
    case Mips::T2   : case Mips::F10: case Mips::D5: return 10;
    case Mips::T3   : case Mips::F11: return 11;
    case Mips::T4   : case Mips::F12: case Mips::D6: return 12;
    case Mips::T5   : case Mips::F13: return 13;
    case Mips::T6   : case Mips::F14: case Mips::D7: return 14;
    case Mips::T7   : case Mips::F15: return 15;
    case Mips::S0   : case Mips::F16: case Mips::D8: return 16;
    case Mips::S1   : case Mips::F17: return 17;
    case Mips::S2   : case Mips::F18: case Mips::D9: return 18;
    case Mips::S3   : case Mips::F19: return 19;
    case Mips::S4   : case Mips::F20: case Mips::D10: return 20;
    case Mips::S5   : case Mips::F21: return 21;
    case Mips::S6   : case Mips::F22: case Mips::D11: return 22;
    case Mips::S7   : case Mips::F23: return 23;
    case Mips::T8   : case Mips::F24: case Mips::D12: return 24;
    case Mips::T9   : case Mips::F25: return 25;
    case Mips::K0   : case Mips::F26: case Mips::D13: return 26;
    case Mips::K1   : case Mips::F27: return 27;
    case Mips::GP   : case Mips::F28: case Mips::D14: return 28;
    case Mips::SP   : case Mips::F29: return 29;
    case Mips::FP   : case Mips::F30: case Mips::D15: return 30;
    case Mips::RA   : case Mips::F31: return 31;
    default: llvm_unreachable("Unknown register number!");
  }
  return 0; // Not reached
}

unsigned MipsRegisterInfo::getPICCallReg() { return Mips::T9; }

//===----------------------------------------------------------------------===//
// Callee Saved Registers methods
//===----------------------------------------------------------------------===//

/// Mips Callee Saved Registers
const unsigned* MipsRegisterInfo::
getCalleeSavedRegs(const MachineFunction *MF) const
{
  // Mips callee-save register range is $16-$23, $f20-$f30
  static const unsigned SingleFloatOnlyCalleeSavedRegs[] = {
    Mips::F31, Mips::F30, Mips::F29, Mips::F28, Mips::F27, Mips::F26,
    Mips::F25, Mips::F24, Mips::F23, Mips::F22, Mips::F21, Mips::F20,
    Mips::RA, Mips::FP, Mips::S7, Mips::S6, Mips::S5, Mips::S4,
    Mips::S3, Mips::S2, Mips::S1, Mips::S0, 0
  };

  static const unsigned Mips32CalleeSavedRegs[] = {
    Mips::D15, Mips::D14, Mips::D13, Mips::D12, Mips::D11, Mips::D10,
    Mips::RA, Mips::FP, Mips::S7, Mips::S6, Mips::S5, Mips::S4,
    Mips::S3, Mips::S2, Mips::S1, Mips::S0, 0
  };

  static const unsigned N32CalleeSavedRegs[] = {
    Mips::D31_64, Mips::D29_64, Mips::D27_64, Mips::D25_64, Mips::D23_64,
    Mips::D21_64,
    Mips::RA_64, Mips::FP_64, Mips::GP_64, Mips::S7_64, Mips::S6_64,
    Mips::S5_64, Mips::S4_64, Mips::S3_64, Mips::S2_64, Mips::S1_64,
    Mips::S0_64, 0
  };

  static const unsigned N64CalleeSavedRegs[] = {
    Mips::D31_64, Mips::D30_64, Mips::D29_64, Mips::D28_64, Mips::D27_64,
    Mips::D26_64, Mips::D25_64, Mips::D24_64,
    Mips::RA_64, Mips::FP_64, Mips::GP_64, Mips::S7_64, Mips::S6_64,
    Mips::S5_64, Mips::S4_64, Mips::S3_64, Mips::S2_64, Mips::S1_64,
    Mips::S0_64, 0
  };

  if (Subtarget.isSingleFloat())
    return SingleFloatOnlyCalleeSavedRegs;
  else if (!Subtarget.hasMips64())
    return Mips32CalleeSavedRegs;
  else if (Subtarget.isABI_N32())
    return N32CalleeSavedRegs;
  
  assert(Subtarget.isABI_N64());
  return N64CalleeSavedRegs;  
}

BitVector MipsRegisterInfo::
getReservedRegs(const MachineFunction &MF) const {
  static const unsigned ReservedCPURegs[] = {
    Mips::ZERO, Mips::AT, Mips::K0, Mips::K1, 
    Mips::GP, Mips::SP, Mips::FP, Mips::RA, 0
  };

  static const unsigned ReservedCPU64Regs[] = {
    Mips::ZERO_64, Mips::AT_64, Mips::K0_64, Mips::K1_64, 
    Mips::GP_64, Mips::SP_64, Mips::FP_64, Mips::RA_64, 0
  };

  BitVector Reserved(getNumRegs());
  typedef TargetRegisterClass::iterator RegIter;

  for (const unsigned *Reg = ReservedCPURegs; *Reg; ++Reg)
    Reserved.set(*Reg);

  if (Subtarget.hasMips64()) {
    for (const unsigned *Reg = ReservedCPU64Regs; *Reg; ++Reg)
      Reserved.set(*Reg);

    // Reserve all registers in AFGR64.
    for (RegIter Reg = Mips::AFGR64RegisterClass->begin();
         Reg != Mips::AFGR64RegisterClass->end(); ++Reg)
      Reserved.set(*Reg);
  }
  else {
    // Reserve all registers in CPU64Regs & FGR64.
    for (RegIter Reg = Mips::CPU64RegsRegisterClass->begin();
         Reg != Mips::CPU64RegsRegisterClass->end(); ++Reg)
      Reserved.set(*Reg);

    for (RegIter Reg = Mips::FGR64RegisterClass->begin();
         Reg != Mips::FGR64RegisterClass->end(); ++Reg)
      Reserved.set(*Reg);
  }
  
  return Reserved;
}

// This function eliminate ADJCALLSTACKDOWN,
// ADJCALLSTACKUP pseudo instructions
void MipsRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  // Simply discard ADJCALLSTACKDOWN, ADJCALLSTACKUP instructions.
  MBB.erase(I);
}

// FrameIndex represent objects inside a abstract stack.
// We must replace FrameIndex with an stack/frame pointer
// direct reference.
void MipsRegisterInfo::
eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                    RegScavenger *RS) const {
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();

  unsigned i = 0;
  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() &&
           "Instr doesn't have FrameIndex operand!");
  }

  DEBUG(errs() << "\nFunction : " << MF.getFunction()->getName() << "\n";
        errs() << "<--------->\n" << MI);

  int FrameIndex = MI.getOperand(i).getIndex();
  int stackSize  = MF.getFrameInfo()->getStackSize();
  int spOffset   = MF.getFrameInfo()->getObjectOffset(FrameIndex);

  DEBUG(errs() << "FrameIndex : " << FrameIndex << "\n"
               << "spOffset   : " << spOffset << "\n"
               << "stackSize  : " << stackSize << "\n");

  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  int MinCSFI = 0;
  int MaxCSFI = -1;

  if (CSI.size()) {
    MinCSFI = CSI[0].getFrameIdx();
    MaxCSFI = CSI[CSI.size() - 1].getFrameIdx();
  }

  // The following stack frame objects are always referenced relative to $sp:
  //  1. Outgoing arguments.
  //  2. Pointer to dynamically allocated stack space.
  //  3. Locations for callee-saved registers.
  // Everything else is referenced relative to whatever register 
  // getFrameRegister() returns.
  unsigned FrameReg;

  if (MipsFI->isOutArgFI(FrameIndex) || MipsFI->isDynAllocFI(FrameIndex) ||
      (FrameIndex >= MinCSFI && FrameIndex <= MaxCSFI))
    FrameReg = Mips::SP;
  else
    FrameReg = getFrameRegister(MF); 
  
  // Calculate final offset.
  // - There is no need to change the offset if the frame object is one of the
  //   following: an outgoing argument, pointer to a dynamically allocated
  //   stack space or a $gp restore location,
  // - If the frame object is any of the following, its offset must be adjusted
  //   by adding the size of the stack:
  //   incoming argument, callee-saved register location or local variable.  
  int Offset;

  if (MipsFI->isOutArgFI(FrameIndex) || MipsFI->isGPFI(FrameIndex) ||
      MipsFI->isDynAllocFI(FrameIndex))
    Offset = spOffset;
  else
    Offset = spOffset + stackSize;

  Offset    += MI.getOperand(i+1).getImm();

  DEBUG(errs() << "Offset     : " << Offset << "\n" << "<--------->\n");

  // If MI is not a debug value, make sure Offset fits in the 16-bit immediate
  // field. 
  if (!MI.isDebugValue() && (Offset >= 0x8000 || Offset < -0x8000)) {
    MachineBasicBlock &MBB = *MI.getParent();
    DebugLoc DL = II->getDebugLoc();
    int ImmHi = (((unsigned)Offset & 0xffff0000) >> 16) +
                ((Offset & 0x8000) != 0);

    // FIXME: change this when mips goes MC".
    BuildMI(MBB, II, DL, TII.get(Mips::NOAT));
    BuildMI(MBB, II, DL, TII.get(Mips::LUi), Mips::AT).addImm(ImmHi);
    BuildMI(MBB, II, DL, TII.get(Mips::ADDu), Mips::AT).addReg(FrameReg)
                                                       .addReg(Mips::AT);
    FrameReg = Mips::AT;
    Offset = (short)(Offset & 0xffff);

    BuildMI(MBB, ++II, MI.getDebugLoc(), TII.get(Mips::ATMACRO));
  }

  MI.getOperand(i).ChangeToRegister(FrameReg, false);
  MI.getOperand(i+1).ChangeToImmediate(Offset);
}

unsigned MipsRegisterInfo::
getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  return TFI->hasFP(MF) ? Mips::FP : Mips::SP;
}

unsigned MipsRegisterInfo::
getEHExceptionRegister() const {
  llvm_unreachable("What is the exception register");
  return 0;
}

unsigned MipsRegisterInfo::
getEHHandlerRegister() const {
  llvm_unreachable("What is the exception handler register");
  return 0;
}
