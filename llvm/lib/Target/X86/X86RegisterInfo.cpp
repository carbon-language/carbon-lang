//===- X86RegisterInfo.cpp - X86 Register Information -----------*- C++ -*-===//
//
// This file contains the X86 implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86RegisterInfo.h"
#include "X86InstrBuilder.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

// X86Regs - Turn the X86RegisterInfo.def file into a bunch of register
// descriptors
//
static const MRegisterDesc X86Regs[] = {
#define R(ENUM, NAME, FLAGS, TSFLAGS) { NAME, FLAGS, TSFLAGS },
#include "X86RegisterInfo.def"
};

X86RegisterInfo::X86RegisterInfo()
  : MRegisterInfo(X86Regs, sizeof(X86Regs)/sizeof(X86Regs[0])) {
}


MachineBasicBlock::iterator
X86RegisterInfo::storeReg2RegOffset(MachineBasicBlock *MBB,
                                    MachineBasicBlock::iterator &MBBI,
                                    unsigned SrcReg, unsigned DestReg, 
                                    unsigned ImmOffset, unsigned dataSize)
  const
{
  MachineInstr *MI = addRegOffset(BuildMI(X86::MOVrm32, 5),
                                  DestReg, ImmOffset).addReg(SrcReg);
  return ++(MBB->insert(MBBI, MI));
}

MachineBasicBlock::iterator
X86RegisterInfo::loadRegOffset2Reg(MachineBasicBlock *MBB,
                                   MachineBasicBlock::iterator &MBBI,
                                   unsigned DestReg, unsigned SrcReg,
                                   unsigned ImmOffset, unsigned dataSize)
  const
{
  MachineInstr *MI = addRegOffset(BuildMI(X86::MOVmr32, 5).addReg(DestReg),
                                  SrcReg, ImmOffset);
  return ++(MBB->insert(MBBI, MI));
}


unsigned X86RegisterInfo::getFramePointer() const {
  return X86::EBP;
}

unsigned X86RegisterInfo::getStackPointer() const {
  return X86::ESP;
}
