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
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/BitVector.h"

using namespace llvm;

// FIXME: Provide proper call frame setup / destroy opcodes.
MSP430RegisterInfo::MSP430RegisterInfo(const TargetInstrInfo &tii)
  : MSP430GenRegisterInfo(MSP430::NOP, MSP430::NOP),
    TII(tii) {}

const unsigned*
MSP430RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  assert(0 && "Not implemented yet!");
}

const TargetRegisterClass* const*
MSP430RegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  assert(0 && "Not implemented yet!");
}

BitVector
MSP430RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());

  // Mark 4 special registers as reserved.
  Reserved.set(MSP430::PC);
  Reserved.set(MSP430::SP);
  Reserved.set(MSP430::SR);
  Reserved.set(MSP430::CG);

  // Mark frame pointer as reserved if needed.
  if (hasFP(MF))
    Reserved.set(MSP430::FP);

  return Reserved;
}

bool MSP430RegisterInfo::hasFP(const MachineFunction &MF) const {
  assert(0 && "Not implemented yet!");
}

void
MSP430RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                        int SPAdj, RegScavenger *RS) const {
  assert(0 && "Not implemented yet!");
}

void MSP430RegisterInfo::emitPrologue(MachineFunction &MF) const {
  assert(0 && "Not implemented yet!");
}

void MSP430RegisterInfo::emitEpilogue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {
  assert(0 && "Not implemented yet!");
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
