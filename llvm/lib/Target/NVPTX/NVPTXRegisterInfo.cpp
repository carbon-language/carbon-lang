//===- NVPTXRegisterInfo.cpp - NVPTX Register Information -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the NVPTX implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "nvptx-reg-info"

#include "NVPTXRegisterInfo.h"
#include "NVPTX.h"
#include "NVPTXSubtarget.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Target/TargetInstrInfo.h"

using namespace llvm;

namespace llvm {
std::string getNVPTXRegClassName(TargetRegisterClass const *RC) {
  if (RC == &NVPTX::Float32RegsRegClass) {
    return ".f32";
  }
  if (RC == &NVPTX::Float64RegsRegClass) {
    return ".f64";
  } else if (RC == &NVPTX::Int64RegsRegClass) {
    return ".s64";
  } else if (RC == &NVPTX::Int32RegsRegClass) {
    return ".s32";
  } else if (RC == &NVPTX::Int16RegsRegClass) {
    return ".s16";
  }
      // Int8Regs become 16-bit registers in PTX
      else if (RC == &NVPTX::Int8RegsRegClass) {
    return ".s16";
  } else if (RC == &NVPTX::Int1RegsRegClass) {
    return ".pred";
  } else if (RC == &NVPTX::SpecialRegsRegClass) {
    return "!Special!";
  } else {
    return "INTERNAL";
  }
  return "";
}

std::string getNVPTXRegClassStr(TargetRegisterClass const *RC) {
  if (RC == &NVPTX::Float32RegsRegClass) {
    return "%f";
  }
  if (RC == &NVPTX::Float64RegsRegClass) {
    return "%fl";
  } else if (RC == &NVPTX::Int64RegsRegClass) {
    return "%rl";
  } else if (RC == &NVPTX::Int32RegsRegClass) {
    return "%r";
  } else if (RC == &NVPTX::Int16RegsRegClass) {
    return "%rs";
  } else if (RC == &NVPTX::Int8RegsRegClass) {
    return "%rc";
  } else if (RC == &NVPTX::Int1RegsRegClass) {
    return "%p";
  } else if (RC == &NVPTX::SpecialRegsRegClass) {
    return "!Special!";
  } else {
    return "INTERNAL";
  }
  return "";
}
}

NVPTXRegisterInfo::NVPTXRegisterInfo(const NVPTXSubtarget &st)
    : NVPTXGenRegisterInfo(0), Is64Bit(st.is64Bit()) {}

#define GET_REGINFO_TARGET_DESC
#include "NVPTXGenRegisterInfo.inc"

/// NVPTX Callee Saved Registers
const uint16_t *
NVPTXRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const uint16_t CalleeSavedRegs[] = { 0 };
  return CalleeSavedRegs;
}

// NVPTX Callee Saved Reg Classes
const TargetRegisterClass *const *
NVPTXRegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  static const TargetRegisterClass *const CalleeSavedRegClasses[] = { 0 };
  return CalleeSavedRegClasses;
}

BitVector NVPTXRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  return Reserved;
}

void NVPTXRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                            int SPAdj, unsigned FIOperandNum,
                                            RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected");

  MachineInstr &MI = *II;
  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();

  MachineFunction &MF = *MI.getParent()->getParent();
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) +
               MI.getOperand(FIOperandNum + 1).getImm();

  // Using I0 as the frame pointer
  MI.getOperand(FIOperandNum).ChangeToRegister(NVPTX::VRFrame, false);
  MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
}

int NVPTXRegisterInfo::getDwarfRegNum(unsigned RegNum, bool isEH) const {
  return 0;
}

unsigned NVPTXRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return NVPTX::VRFrame;
}

unsigned NVPTXRegisterInfo::getRARegister() const { return 0; }
