//===- MSP430RegisterInfo.cpp - MSP430 Register Information ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MSP430 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "msp430-reg-info"

#include "MSP430.h"
#include "MSP430RegisterInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/BitVector.h"

using namespace llvm;

// FIXME: Provide proper call frame setup / destroy opcodes.
MSP430RegisterInfo::MSP430RegisterInfo(const TargetInstrInfo &tii)
  : MSP430GenRegisterInfo(MSP430::NOP, MSP430::NOP),
    TII(tii) {}

const unsigned*
MSP430RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const unsigned CalleeSavedRegs[] = {
    MSP430::FPW, MSP430::R5W, MSP430::R6W, MSP430::R7W,
    MSP430::R8W, MSP430::R9W, MSP430::R10W, MSP430::R11W,
    0
  };

  return CalleeSavedRegs;
}

const TargetRegisterClass* const*
MSP430RegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  static const TargetRegisterClass * const CalleeSavedRegClasses[] = {
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    0
  };

  return CalleeSavedRegClasses;
}

BitVector
MSP430RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());

  // Mark 4 special registers as reserved.
  Reserved.set(MSP430::PCW);
  Reserved.set(MSP430::SPW);
  Reserved.set(MSP430::SRW);
  Reserved.set(MSP430::CGW);

  // Mark frame pointer as reserved if needed.
  if (hasFP(MF))
    Reserved.set(MSP430::FPW);

  return Reserved;
}

bool MSP430RegisterInfo::hasFP(const MachineFunction &MF) const {
  return NoFramePointerElim || MF.getFrameInfo()->hasVarSizedObjects();
}

void
MSP430RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                        int SPAdj, RegScavenger *RS) const {
  assert(0 && "Not implemented yet!");
}

void MSP430RegisterInfo::emitPrologue(MachineFunction &MF) const {
  // Nothing here yet
}

void MSP430RegisterInfo::emitEpilogue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {
  // Nothing here yet
}

unsigned MSP430RegisterInfo::getRARegister() const {
  assert(0 && "Not implemented yet!");
}

unsigned MSP430RegisterInfo::getFrameRegister(MachineFunction &MF) const {
  assert(0 && "Not implemented yet!");
}

int MSP430RegisterInfo::getDwarfRegNum(unsigned RegNum, bool isEH) const {
  assert(0 && "Not implemented yet!");
}

#include "MSP430GenRegisterInfo.inc"
