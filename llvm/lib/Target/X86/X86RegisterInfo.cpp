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
#include "llvm/CodeGen/MachineFunction.h"

// Create static lists to contain register alias sets...
#define ALIASLIST(NAME, ...) \
  static const unsigned NAME[] = { __VA_ARGS__ };
#include "X86RegisterInfo.def"


// X86Regs - Turn the X86RegisterInfo.def file into a bunch of register
// descriptors
//
static const MRegisterDesc X86Regs[] = {
#define R(ENUM, NAME, FLAGS, TSFLAGS, ALIAS_SET) \
         { NAME, ALIAS_SET, FLAGS, TSFLAGS },
#include "X86RegisterInfo.def"
};

X86RegisterInfo::X86RegisterInfo()
  : MRegisterInfo(X86Regs, sizeof(X86Regs)/sizeof(X86Regs[0])) {
}

unsigned getIdx(unsigned dataSize) {
  switch (dataSize) {
  case 1: return 0;
  case 2: return 1;
  case 4: return 2;
    // FIXME: longs handled as ints
  case 8: return 2;
  default: assert(0 && "Invalid data size!");
  }
}

MachineBasicBlock::iterator
X86RegisterInfo::storeReg2RegOffset(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI,
                                    unsigned SrcReg, unsigned DestReg, 
                                    unsigned ImmOffset, unsigned dataSize)
  const
{
  static const unsigned Opcode[] = { X86::MOVrm8, X86::MOVrm16, X86::MOVrm32 };
  MachineInstr *MI = addRegOffset(BuildMI(Opcode[getIdx(dataSize)], 5),
                                  DestReg, ImmOffset).addReg(SrcReg);
  return ++MBB.insert(MBBI, MI);
}

MachineBasicBlock::iterator
X86RegisterInfo::loadRegOffset2Reg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   unsigned DestReg, unsigned SrcReg,
                                   unsigned ImmOffset, unsigned dataSize)
  const
{
  static const unsigned Opcode[] = { X86::MOVmr8, X86::MOVmr16, X86::MOVmr32 };
  MachineInstr *MI = addRegOffset(BuildMI(Opcode[getIdx(dataSize)], 4, DestReg),
                                  SrcReg, ImmOffset);
  return ++MBB.insert(MBBI, MI);
}

MachineBasicBlock::iterator
X86RegisterInfo::moveReg2Reg(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MBBI,
                             unsigned DestReg, unsigned SrcReg,
                             unsigned dataSize) const
{
  static const unsigned Opcode[] = { X86::MOVrr8, X86::MOVrr16, X86::MOVrr32 };
  MachineInstr *MI = BuildMI(Opcode[getIdx(dataSize)],1,DestReg).addReg(SrcReg);
  return ++MBB.insert(MBBI, MI);
}

MachineBasicBlock::iterator
X86RegisterInfo::moveImm2Reg(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MBBI,
                             unsigned DestReg, unsigned Imm, unsigned dataSize)
  const
{
  static const unsigned Opcode[] = { X86::MOVir8, X86::MOVir16, X86::MOVir32 };
  MachineInstr *MI = BuildMI(Opcode[getIdx(dataSize)], 1, DestReg).addReg(Imm);
  return ++MBB.insert(MBBI, MI);
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

void X86RegisterInfo::emitPrologue(MachineFunction &MF,
                                   unsigned numBytes) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineBasicBlock::iterator MBBI = MBB.begin();

  // PUSH all callee-save registers
  const unsigned* regs = getCalleeSaveRegs();
  while (*regs) {
    MachineInstr *MI = BuildMI(X86::PUSHr32, 1).addReg(*regs);
    MBBI = ++MBB.insert(MBBI, MI);
    ++regs;
  }

  // PUSH ebp
  MachineInstr *MI = BuildMI(X86::PUSHr32, 1).addReg(X86::EBP);
  MBBI = ++MBB.insert(MBBI, MI);

  // MOV ebp, esp
  MI = BuildMI(X86::MOVrr32, 1, X86::EBP).addReg(X86::ESP);
  MBBI = ++MBB.insert(MBBI, MI);

  // adjust stack pointer: ESP -= numbytes
  MI  = BuildMI(X86::SUBri32, 2, X86::ESP).addReg(X86::ESP).addZImm(numBytes);
  MBBI = ++MBB.insert(MBBI, MI);
}

void X86RegisterInfo::emitEpilogue(MachineBasicBlock &MBB,
                                   unsigned numBytes) const {
  MachineBasicBlock::iterator MBBI = --MBB.end();
  assert((*MBBI)->getOpcode() == X86::RET &&
         "Can only insert epilog into returning blocks");

  // insert LEAVE
  MBBI = ++MBB.insert(MBBI, BuildMI(X86::LEAVE, 0));

  // POP all callee-save registers in REVERSE ORDER
  static const unsigned regs[] = { X86::EBX, X86::EDI, X86::ESI,
                                   MRegisterInfo::NoRegister };
  unsigned idx = 0;
  while (regs[idx]) {
    MachineInstr *MI = BuildMI(X86::POPr32, 0, regs[idx++]);
    MBBI = ++(MBB.insert(MBBI, MI));
  }
}
