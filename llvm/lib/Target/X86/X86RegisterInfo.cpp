//===- X86RegisterInfo.cpp - X86 Register Information ---------------------===//
//
// This file contains the X86 implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86RegisterInfo.h"
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


void X86RegisterInfo::copyReg2PCRel(MachineBasicBlock *MBB,
                                    MachineBasicBlock::iterator &MBBI,
                                    unsigned SrcReg, unsigned ImmOffset,
                                    unsigned dataSize) const
{
  MachineInstrBuilder MI = BuildMI(X86::MOVmr32, 2)
    .addPCDisp(ConstantUInt::get(Type::UIntTy, ImmOffset)).addReg(SrcReg);
  MBB->insert(MBBI, &*MI);
}

void X86RegisterInfo::copyPCRel2Reg(MachineBasicBlock *MBB,
                                    MachineBasicBlock::iterator &MBBI,
                                    unsigned ImmOffset, unsigned DestReg,
                                    unsigned dataSize) const
{
  MachineInstrBuilder MI = BuildMI(X86::MOVrm32, 2)
    .addReg(DestReg).addPCDisp(ConstantUInt::get(Type::UIntTy, ImmOffset));
  MBB->insert(MBBI, &*MI);
}

