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
                                    MachineBasicBlock::iterator MBBI,
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
                                   MachineBasicBlock::iterator MBBI,
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

const unsigned* X86RegisterInfo::getCalleeSaveRegs() const {
  static const unsigned CalleeSaveRegs[] = { X86::ESI, X86::EDI, X86::EBX,
                                             MRegisterInfo::NoRegister };
  return CalleeSaveRegs;
}


const unsigned* X86RegisterInfo::getCallerSaveRegs() const {
  static const unsigned CallerSaveRegs[] = { X86::EAX, X86::ECX, X86::EDX,
                                             MRegisterInfo::NoRegister };
  return CallerSaveRegs;
}

MachineBasicBlock::iterator 
X86RegisterInfo::emitPrologue(MachineBasicBlock *MBB,
                              MachineBasicBlock::iterator MBBI,
                              unsigned numBytes) const
{
  MachineInstr *MI;

  // PUSH ebp
  MI = BuildMI (X86::PUSHr32, 1).addReg(X86::EBP);
  MBBI = ++(MBB->insert(MBBI, MI));

  // MOV ebp, esp
  MI = BuildMI (X86::MOVrr32, 2).addReg(X86::EBP).addReg(X86::ESP);
  MBBI = ++(MBB->insert(MBBI, MI));  

  // adjust stack pointer
  MI  = BuildMI(X86::SUBri32, 2).addReg(X86::ESP).addZImm(numBytes);
  MBBI = ++(MBB->insert(MBBI, MI));

  // PUSH all callee-save registers
  const unsigned* regs = getCalleeSaveRegs();
  while (*regs) {
    MI = BuildMI(X86::PUSHr32, 1).addReg(*regs);
    MBBI = ++(MBB->insert(MBBI, MI));
    ++regs;
  }

  return MBBI;
}

MachineBasicBlock::iterator
X86RegisterInfo::emitEpilogue(MachineBasicBlock *MBB,
                              MachineBasicBlock::iterator MBBI,
                              unsigned numBytes) const
{
  MachineInstr *MI;

  // POP all callee-save registers in REVERSE ORDER
  static const unsigned regs[] = { X86::EBX, X86::EDI, X86::ESI,
                                   MRegisterInfo::NoRegister };
  unsigned idx = 0;
  while (regs[idx]) {
    MI = BuildMI(X86::POPr32, 1).addReg(regs[idx++]);
    MBBI = ++(MBB->insert(MBBI, MI));
  }
  
  // insert LEAVE
  MI = BuildMI(X86::LEAVE, 0);
  MBBI = ++(MBB->insert(MBBI, MI));
  
  return MBBI;
}
