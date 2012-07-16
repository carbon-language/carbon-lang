//===-- AMDGPUInstrInfo.cpp - Base class for AMD GPU InstrInfo ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the TargetInstrInfo class that is
// common to all AMD GPUs.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUInstrInfo.h"
#include "AMDGPURegisterInfo.h"
#include "AMDGPUTargetMachine.h"
#include "AMDIL.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

AMDGPUInstrInfo::AMDGPUInstrInfo(AMDGPUTargetMachine &tm)
  : AMDILInstrInfo(tm) { }

void AMDGPUInstrInfo::convertToISA(MachineInstr & MI, MachineFunction &MF,
    DebugLoc DL) const
{
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const AMDGPURegisterInfo & RI = getRegisterInfo();

  for (unsigned i = 0; i < MI.getNumOperands(); i++) {
    MachineOperand &MO = MI.getOperand(i);
    // Convert dst regclass to one that is supported by the ISA
    if (MO.isReg() && MO.isDef()) {
      if (TargetRegisterInfo::isVirtualRegister(MO.getReg())) {
        const TargetRegisterClass * oldRegClass = MRI.getRegClass(MO.getReg());
        const TargetRegisterClass * newRegClass = RI.getISARegClass(oldRegClass);

        assert(newRegClass);

        MRI.setRegClass(MO.getReg(), newRegClass);
      }
    }
  }
}
