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

static unsigned getIdx(const TargetRegisterClass *RC) {
  switch (RC->getDataSize()) {
  default: assert(0 && "Invalid data size!");
  case 1:  return 0;
  case 2:  return 1;
  case 4:  return 2;
  case 10: return 3;
  }
}

void X86RegisterInfo::storeReg2RegOffset(MachineBasicBlock &MBB,
					 MachineBasicBlock::iterator &MBBI,
					 unsigned SrcReg, unsigned DestReg, 
					 unsigned ImmOffset,
					 const TargetRegisterClass *RC) const {
  static const unsigned Opcode[] =
    { X86::MOVrm8, X86::MOVrm16, X86::MOVrm32, X86::FSTPr80 };
  MachineInstr *MI = addRegOffset(BuildMI(Opcode[getIdx(RC)], 5),
                                  DestReg, ImmOffset).addReg(SrcReg);
  MBBI = MBB.insert(MBBI, MI)+1;
}

void X86RegisterInfo::loadRegOffset2Reg(MachineBasicBlock &MBB,
					MachineBasicBlock::iterator &MBBI,
					unsigned DestReg, unsigned SrcReg,
					unsigned ImmOffset,
					const TargetRegisterClass *RC) const {
  static const unsigned Opcode[] =
    { X86::MOVmr8, X86::MOVmr16, X86::MOVmr32, X86::FLDr80 };
  MachineInstr *MI = addRegOffset(BuildMI(Opcode[getIdx(RC)], 4, DestReg),
                                  SrcReg, ImmOffset);
  MBBI = MBB.insert(MBBI, MI)+1;
}

void X86RegisterInfo::moveReg2Reg(MachineBasicBlock &MBB,
				  MachineBasicBlock::iterator &MBBI,
				  unsigned DestReg, unsigned SrcReg,
				  const TargetRegisterClass *RC) const {
  static const unsigned Opcode[] =
    { X86::MOVrr8, X86::MOVrr16, X86::MOVrr32, X86::FpMOV };
  MachineInstr *MI = BuildMI(Opcode[getIdx(RC)],1,DestReg).addReg(SrcReg);
  MBBI = MBB.insert(MBBI, MI)+1;
}

void X86RegisterInfo::moveImm2Reg(MachineBasicBlock &MBB,
				  MachineBasicBlock::iterator &MBBI,
				  unsigned DestReg, unsigned Imm,
				  const TargetRegisterClass *RC) const {
  static const unsigned Opcode[] =
    { X86::MOVir8, X86::MOVir16, X86::MOVir32, 0 };
  MachineInstr *MI = BuildMI(Opcode[getIdx(RC)], 1, DestReg).addReg(Imm);
  assert(MI->getOpcode() != 0 && "Cannot move FP imm to reg yet!");
  MBBI = MBB.insert(MBBI, MI)+1;
}


unsigned X86RegisterInfo::getFramePointer() const {
  return X86::EBP;
}

unsigned X86RegisterInfo::getStackPointer() const {
  return X86::ESP;
}

const unsigned* X86RegisterInfo::getCalleeSaveRegs() const {
  static const unsigned CalleeSaveRegs[] = { X86::ESI, X86::EDI, X86::EBX, 0 };
  return CalleeSaveRegs;
}


const unsigned* X86RegisterInfo::getCallerSaveRegs() const {
  static const unsigned CallerSaveRegs[] = { X86::EAX, X86::ECX, X86::EDX, 0 };
  return CallerSaveRegs;
}

void X86RegisterInfo::emitPrologue(MachineFunction &MF,
                                   unsigned NumBytes) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineBasicBlock::iterator MBBI = MBB.begin();

  // Round stack allocation up to a nice alignment to keep the stack aligned
  NumBytes = (NumBytes + 3) & ~3;

  // PUSH ebp
  MachineInstr *MI = BuildMI(X86::PUSHr32, 1).addReg(X86::EBP);
  MBBI = MBB.insert(MBBI, MI)+1;

  // MOV ebp, esp
  MI = BuildMI(X86::MOVrr32, 1, X86::EBP).addReg(X86::ESP);
  MBBI = MBB.insert(MBBI, MI)+1;

  // adjust stack pointer: ESP -= numbytes
  MI  = BuildMI(X86::SUBri32, 2, X86::ESP).addReg(X86::ESP).addZImm(NumBytes);
  MBBI = 1+MBB.insert(MBBI, MI);
}

void X86RegisterInfo::emitEpilogue(MachineBasicBlock &MBB,
                                   unsigned numBytes) const {
  MachineBasicBlock::iterator MBBI = MBB.end()-1;
  assert((*MBBI)->getOpcode() == X86::RET &&
         "Can only insert epilog into returning blocks");

  // insert LEAVE: mov ESP, EBP; pop EBP
  MBBI = 1+MBB.insert(MBBI, BuildMI(X86::MOVrr32, 1,X86::ESP).addReg(X86::EBP));
  MBBI = 1+MBB.insert(MBBI, BuildMI(X86::POPr32, 1).addReg(X86::EBP));
}
