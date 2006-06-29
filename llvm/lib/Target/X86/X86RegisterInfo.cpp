//===- X86RegisterInfo.cpp - X86 Register Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the MRegisterInfo class.  This
// file is responsible for the frame pointer elimination optimization on X86.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "X86InstrBuilder.h"
#include "X86MachineFunctionInfo.h"
#include "X86TargetMachine.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/STLExtras.h"
#include <iostream>

using namespace llvm;

namespace {
  cl::opt<bool>
  NoFusing("disable-spill-fusing",
           cl::desc("Disable fusing of spill code into instructions"));
  cl::opt<bool>
  PrintFailedFusing("print-failed-fuse-candidates",
                    cl::desc("Print instructions that the allocator wants to"
                             " fuse, but the X86 backend currently can't"),
                    cl::Hidden);
}

X86RegisterInfo::X86RegisterInfo()
  : X86GenRegisterInfo(X86::ADJCALLSTACKDOWN, X86::ADJCALLSTACKUP) {}

void X86RegisterInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MI,
                                          unsigned SrcReg, int FrameIdx,
                                          const TargetRegisterClass *RC) const {
  unsigned Opc;
  if (RC == &X86::GR32RegClass) {
    Opc = X86::MOV32mr;
  } else if (RC == &X86::GR16RegClass) {
    Opc = X86::MOV16mr;
  } else if (RC == &X86::GR8RegClass) {
    Opc = X86::MOV8mr;
  } else if (RC == &X86::GR32_RegClass) {
    Opc = X86::MOV32_mr;
  } else if (RC == &X86::GR16_RegClass) {
    Opc = X86::MOV16_mr;
  } else if (RC == &X86::RFPRegClass || RC == &X86::RSTRegClass) {
    Opc = X86::FpST64m;
  } else if (RC == &X86::FR32RegClass) {
    Opc = X86::MOVSSmr;
  } else if (RC == &X86::FR64RegClass) {
    Opc = X86::MOVSDmr;
  } else if (RC == &X86::VR128RegClass) {
    Opc = X86::MOVAPSmr;
  } else {
    assert(0 && "Unknown regclass");
    abort();
  }
  addFrameReference(BuildMI(MBB, MI, Opc, 5), FrameIdx).addReg(SrcReg);
}

void X86RegisterInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MI,
                                           unsigned DestReg, int FrameIdx,
                                           const TargetRegisterClass *RC) const{
  unsigned Opc;
  if (RC == &X86::GR32RegClass) {
    Opc = X86::MOV32rm;
  } else if (RC == &X86::GR16RegClass) {
    Opc = X86::MOV16rm;
  } else if (RC == &X86::GR8RegClass) {
    Opc = X86::MOV8rm;
  } else if (RC == &X86::GR32_RegClass) {
    Opc = X86::MOV32_rm;
  } else if (RC == &X86::GR16_RegClass) {
    Opc = X86::MOV16_rm;
  } else if (RC == &X86::RFPRegClass || RC == &X86::RSTRegClass) {
    Opc = X86::FpLD64m;
  } else if (RC == &X86::FR32RegClass) {
    Opc = X86::MOVSSrm;
  } else if (RC == &X86::FR64RegClass) {
    Opc = X86::MOVSDrm;
  } else if (RC == &X86::VR128RegClass) {
    Opc = X86::MOVAPSrm;
  } else {
    assert(0 && "Unknown regclass");
    abort();
  }
  addFrameReference(BuildMI(MBB, MI, Opc, 4, DestReg), FrameIdx);
}

void X86RegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned DestReg, unsigned SrcReg,
                                   const TargetRegisterClass *RC) const {
  unsigned Opc;
  if (RC == &X86::GR32RegClass) {
    Opc = X86::MOV32rr;
  } else if (RC == &X86::GR16RegClass) {
    Opc = X86::MOV16rr;
  } else if (RC == &X86::GR8RegClass) {
    Opc = X86::MOV8rr;
  } else if (RC == &X86::GR32_RegClass) {
    Opc = X86::MOV32_rr;
  } else if (RC == &X86::GR16_RegClass) {
    Opc = X86::MOV16_rr;
  } else if (RC == &X86::RFPRegClass || RC == &X86::RSTRegClass) {
    Opc = X86::FpMOV;
  } else if (RC == &X86::FR32RegClass) {
    Opc = X86::FsMOVAPSrr;
  } else if (RC == &X86::FR64RegClass) {
    Opc = X86::FsMOVAPDrr;
  } else if (RC == &X86::VR128RegClass) {
    Opc = X86::MOVAPSrr;
  } else {
    assert(0 && "Unknown regclass");
    abort();
  }
  BuildMI(MBB, MI, Opc, 1, DestReg).addReg(SrcReg);
}


static MachineInstr *MakeMInst(unsigned Opcode, unsigned FrameIndex,
                               MachineInstr *MI) {
  return addFrameReference(BuildMI(Opcode, 4), FrameIndex);
}

static MachineInstr *MakeMRInst(unsigned Opcode, unsigned FrameIndex,
                                MachineInstr *MI) {
  return addFrameReference(BuildMI(Opcode, 5), FrameIndex)
                 .addReg(MI->getOperand(1).getReg());
}

static MachineInstr *MakeMRIInst(unsigned Opcode, unsigned FrameIndex,
                                 MachineInstr *MI) {
  return addFrameReference(BuildMI(Opcode, 6), FrameIndex)
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImmedValue());
}

static MachineInstr *MakeMIInst(unsigned Opcode, unsigned FrameIndex,
                                MachineInstr *MI) {
  if (MI->getOperand(1).isImmediate())
    return addFrameReference(BuildMI(Opcode, 5), FrameIndex)
      .addImm(MI->getOperand(1).getImmedValue());
  else if (MI->getOperand(1).isGlobalAddress())
    return addFrameReference(BuildMI(Opcode, 5), FrameIndex)
      .addGlobalAddress(MI->getOperand(1).getGlobal(),
                        MI->getOperand(1).getOffset());
  else if (MI->getOperand(1).isJumpTableIndex())
    return addFrameReference(BuildMI(Opcode, 5), FrameIndex)
      .addJumpTableIndex(MI->getOperand(1).getJumpTableIndex());
  assert(0 && "Unknown operand for MakeMI!");
  return 0;
}

static MachineInstr *MakeM0Inst(unsigned Opcode, unsigned FrameIndex,
                                MachineInstr *MI) {
  return addFrameReference(BuildMI(Opcode, 5), FrameIndex).addImm(0);
}

static MachineInstr *MakeRMInst(unsigned Opcode, unsigned FrameIndex,
                                MachineInstr *MI) {
  const MachineOperand& op = MI->getOperand(0);
  return addFrameReference(BuildMI(Opcode, 5, op.getReg(), op.getUseType()),
                           FrameIndex);
}

static MachineInstr *MakeRMIInst(unsigned Opcode, unsigned FrameIndex,
                                 MachineInstr *MI) {
  const MachineOperand& op = MI->getOperand(0);
  return addFrameReference(BuildMI(Opcode, 6, op.getReg(), op.getUseType()),
                        FrameIndex).addImm(MI->getOperand(2).getImmedValue());
}


MachineInstr* X86RegisterInfo::foldMemoryOperand(MachineInstr* MI,
                                                 unsigned i,
                                                 int FrameIndex) const {
  if (NoFusing) return NULL;

  /// FIXME: This should obviously be autogenerated by tablegen when patterns
  /// are available!
  if (i == 0) {
    switch(MI->getOpcode()) {
    case X86::XCHG8rr:   return MakeMRInst(X86::XCHG8mr ,FrameIndex, MI);
    case X86::XCHG16rr:  return MakeMRInst(X86::XCHG16mr,FrameIndex, MI);
    case X86::XCHG32rr:  return MakeMRInst(X86::XCHG32mr,FrameIndex, MI);
    case X86::MOV8rr:    return MakeMRInst(X86::MOV8mr , FrameIndex, MI);
    case X86::MOV16rr:   return MakeMRInst(X86::MOV16mr, FrameIndex, MI);
    case X86::MOV32rr:   return MakeMRInst(X86::MOV32mr, FrameIndex, MI);
    case X86::MOV8ri:    return MakeMIInst(X86::MOV8mi , FrameIndex, MI);
    case X86::MOV16ri:   return MakeMIInst(X86::MOV16mi, FrameIndex, MI);
    case X86::MOV32ri:   return MakeMIInst(X86::MOV32mi, FrameIndex, MI);
    case X86::MUL8r:     return MakeMInst( X86::MUL8m ,  FrameIndex, MI);
    case X86::MUL16r:    return MakeMInst( X86::MUL16m,  FrameIndex, MI);
    case X86::MUL32r:    return MakeMInst( X86::MUL32m,  FrameIndex, MI);
    case X86::IMUL8r:    return MakeMInst( X86::IMUL8m , FrameIndex, MI);
    case X86::IMUL16r:   return MakeMInst( X86::IMUL16m, FrameIndex, MI);
    case X86::IMUL32r:   return MakeMInst( X86::IMUL32m, FrameIndex, MI);
    case X86::DIV8r:     return MakeMInst( X86::DIV8m ,  FrameIndex, MI);
    case X86::DIV16r:    return MakeMInst( X86::DIV16m,  FrameIndex, MI);
    case X86::DIV32r:    return MakeMInst( X86::DIV32m,  FrameIndex, MI);
    case X86::IDIV8r:    return MakeMInst( X86::IDIV8m , FrameIndex, MI);
    case X86::IDIV16r:   return MakeMInst( X86::IDIV16m, FrameIndex, MI);
    case X86::IDIV32r:   return MakeMInst( X86::IDIV32m, FrameIndex, MI);
    case X86::NEG8r:     return MakeMInst( X86::NEG8m ,  FrameIndex, MI);
    case X86::NEG16r:    return MakeMInst( X86::NEG16m,  FrameIndex, MI);
    case X86::NEG32r:    return MakeMInst( X86::NEG32m,  FrameIndex, MI);
    case X86::NOT8r:     return MakeMInst( X86::NOT8m ,  FrameIndex, MI);
    case X86::NOT16r:    return MakeMInst( X86::NOT16m,  FrameIndex, MI);
    case X86::NOT32r:    return MakeMInst( X86::NOT32m,  FrameIndex, MI);
    case X86::INC8r:     return MakeMInst( X86::INC8m ,  FrameIndex, MI);
    case X86::INC16r:    return MakeMInst( X86::INC16m,  FrameIndex, MI);
    case X86::INC32r:    return MakeMInst( X86::INC32m,  FrameIndex, MI);
    case X86::DEC8r:     return MakeMInst( X86::DEC8m ,  FrameIndex, MI);
    case X86::DEC16r:    return MakeMInst( X86::DEC16m,  FrameIndex, MI);
    case X86::DEC32r:    return MakeMInst( X86::DEC32m,  FrameIndex, MI);
    case X86::ADD8rr:    return MakeMRInst(X86::ADD8mr , FrameIndex, MI);
    case X86::ADD16rr:   return MakeMRInst(X86::ADD16mr, FrameIndex, MI);
    case X86::ADD32rr:   return MakeMRInst(X86::ADD32mr, FrameIndex, MI);
    case X86::ADD8ri:    return MakeMIInst(X86::ADD8mi , FrameIndex, MI);
    case X86::ADD16ri:   return MakeMIInst(X86::ADD16mi, FrameIndex, MI);
    case X86::ADD32ri:   return MakeMIInst(X86::ADD32mi, FrameIndex, MI);
    case X86::ADD16ri8:  return MakeMIInst(X86::ADD16mi8,FrameIndex, MI);
    case X86::ADD32ri8:  return MakeMIInst(X86::ADD32mi8,FrameIndex, MI);
    case X86::ADC32rr:   return MakeMRInst(X86::ADC32mr, FrameIndex, MI);
    case X86::ADC32ri:   return MakeMIInst(X86::ADC32mi, FrameIndex, MI);
    case X86::ADC32ri8:  return MakeMIInst(X86::ADC32mi8,FrameIndex, MI);
    case X86::SUB8rr:    return MakeMRInst(X86::SUB8mr , FrameIndex, MI);
    case X86::SUB16rr:   return MakeMRInst(X86::SUB16mr, FrameIndex, MI);
    case X86::SUB32rr:   return MakeMRInst(X86::SUB32mr, FrameIndex, MI);
    case X86::SUB8ri:    return MakeMIInst(X86::SUB8mi , FrameIndex, MI);
    case X86::SUB16ri:   return MakeMIInst(X86::SUB16mi, FrameIndex, MI);
    case X86::SUB32ri:   return MakeMIInst(X86::SUB32mi, FrameIndex, MI);
    case X86::SUB16ri8:  return MakeMIInst(X86::SUB16mi8,FrameIndex, MI);
    case X86::SUB32ri8:  return MakeMIInst(X86::SUB32mi8,FrameIndex, MI);
    case X86::SBB32rr:   return MakeMRInst(X86::SBB32mr, FrameIndex, MI);
    case X86::SBB32ri:   return MakeMIInst(X86::SBB32mi, FrameIndex, MI);
    case X86::SBB32ri8:  return MakeMIInst(X86::SBB32mi8,FrameIndex, MI);
    case X86::AND8rr:    return MakeMRInst(X86::AND8mr , FrameIndex, MI);
    case X86::AND16rr:   return MakeMRInst(X86::AND16mr, FrameIndex, MI);
    case X86::AND32rr:   return MakeMRInst(X86::AND32mr, FrameIndex, MI);
    case X86::AND8ri:    return MakeMIInst(X86::AND8mi , FrameIndex, MI);
    case X86::AND16ri:   return MakeMIInst(X86::AND16mi, FrameIndex, MI);
    case X86::AND32ri:   return MakeMIInst(X86::AND32mi, FrameIndex, MI);
    case X86::AND16ri8:  return MakeMIInst(X86::AND16mi8,FrameIndex, MI);
    case X86::AND32ri8:  return MakeMIInst(X86::AND32mi8,FrameIndex, MI);
    case X86::OR8rr:     return MakeMRInst(X86::OR8mr ,  FrameIndex, MI);
    case X86::OR16rr:    return MakeMRInst(X86::OR16mr,  FrameIndex, MI);
    case X86::OR32rr:    return MakeMRInst(X86::OR32mr,  FrameIndex, MI);
    case X86::OR8ri:     return MakeMIInst(X86::OR8mi ,  FrameIndex, MI);
    case X86::OR16ri:    return MakeMIInst(X86::OR16mi,  FrameIndex, MI);
    case X86::OR32ri:    return MakeMIInst(X86::OR32mi,  FrameIndex, MI);
    case X86::OR16ri8:   return MakeMIInst(X86::OR16mi8, FrameIndex, MI);
    case X86::OR32ri8:   return MakeMIInst(X86::OR32mi8, FrameIndex, MI);
    case X86::XOR8rr:    return MakeMRInst(X86::XOR8mr , FrameIndex, MI);
    case X86::XOR16rr:   return MakeMRInst(X86::XOR16mr, FrameIndex, MI);
    case X86::XOR32rr:   return MakeMRInst(X86::XOR32mr, FrameIndex, MI);
    case X86::XOR8ri:    return MakeMIInst(X86::XOR8mi , FrameIndex, MI);
    case X86::XOR16ri:   return MakeMIInst(X86::XOR16mi, FrameIndex, MI);
    case X86::XOR32ri:   return MakeMIInst(X86::XOR32mi, FrameIndex, MI);
    case X86::XOR16ri8:  return MakeMIInst(X86::XOR16mi8,FrameIndex, MI);
    case X86::XOR32ri8:  return MakeMIInst(X86::XOR32mi8,FrameIndex, MI);
    case X86::SHL8rCL:   return MakeMInst( X86::SHL8mCL ,FrameIndex, MI);
    case X86::SHL16rCL:  return MakeMInst( X86::SHL16mCL,FrameIndex, MI);
    case X86::SHL32rCL:  return MakeMInst( X86::SHL32mCL,FrameIndex, MI);
    case X86::SHL8ri:    return MakeMIInst(X86::SHL8mi , FrameIndex, MI);
    case X86::SHL16ri:   return MakeMIInst(X86::SHL16mi, FrameIndex, MI);
    case X86::SHL32ri:   return MakeMIInst(X86::SHL32mi, FrameIndex, MI);
    case X86::SHL8r1:    return MakeMInst(X86::SHL8m1 , FrameIndex, MI);
    case X86::SHL16r1:   return MakeMInst(X86::SHL16m1, FrameIndex, MI);
    case X86::SHL32r1:   return MakeMInst(X86::SHL32m1, FrameIndex, MI);
    case X86::SHR8rCL:   return MakeMInst( X86::SHR8mCL ,FrameIndex, MI);
    case X86::SHR16rCL:  return MakeMInst( X86::SHR16mCL,FrameIndex, MI);
    case X86::SHR32rCL:  return MakeMInst( X86::SHR32mCL,FrameIndex, MI);
    case X86::SHR8ri:    return MakeMIInst(X86::SHR8mi , FrameIndex, MI);
    case X86::SHR16ri:   return MakeMIInst(X86::SHR16mi, FrameIndex, MI);
    case X86::SHR32ri:   return MakeMIInst(X86::SHR32mi, FrameIndex, MI);
    case X86::SHR8r1:    return MakeMInst(X86::SHR8m1 , FrameIndex, MI);
    case X86::SHR16r1:   return MakeMInst(X86::SHR16m1, FrameIndex, MI);
    case X86::SHR32r1:   return MakeMInst(X86::SHR32m1, FrameIndex, MI);
    case X86::SAR8rCL:   return MakeMInst( X86::SAR8mCL ,FrameIndex, MI);
    case X86::SAR16rCL:  return MakeMInst( X86::SAR16mCL,FrameIndex, MI);
    case X86::SAR32rCL:  return MakeMInst( X86::SAR32mCL,FrameIndex, MI);
    case X86::SAR8ri:    return MakeMIInst(X86::SAR8mi , FrameIndex, MI);
    case X86::SAR16ri:   return MakeMIInst(X86::SAR16mi, FrameIndex, MI);
    case X86::SAR32ri:   return MakeMIInst(X86::SAR32mi, FrameIndex, MI);
    case X86::SAR8r1:    return MakeMInst(X86::SAR8m1 , FrameIndex, MI);
    case X86::SAR16r1:   return MakeMInst(X86::SAR16m1, FrameIndex, MI);
    case X86::SAR32r1:   return MakeMInst(X86::SAR32m1, FrameIndex, MI);
    case X86::ROL8rCL:   return MakeMInst( X86::ROL8mCL ,FrameIndex, MI);
    case X86::ROL16rCL:  return MakeMInst( X86::ROL16mCL,FrameIndex, MI);
    case X86::ROL32rCL:  return MakeMInst( X86::ROL32mCL,FrameIndex, MI);
    case X86::ROL8ri:    return MakeMIInst(X86::ROL8mi , FrameIndex, MI);
    case X86::ROL16ri:   return MakeMIInst(X86::ROL16mi, FrameIndex, MI);
    case X86::ROL32ri:   return MakeMIInst(X86::ROL32mi, FrameIndex, MI);
    case X86::ROL8r1:    return MakeMInst(X86::ROL8m1 , FrameIndex, MI);
    case X86::ROL16r1:   return MakeMInst(X86::ROL16m1, FrameIndex, MI);
    case X86::ROL32r1:   return MakeMInst(X86::ROL32m1, FrameIndex, MI);
    case X86::ROR8rCL:   return MakeMInst( X86::ROR8mCL ,FrameIndex, MI);
    case X86::ROR16rCL:  return MakeMInst( X86::ROR16mCL,FrameIndex, MI);
    case X86::ROR32rCL:  return MakeMInst( X86::ROR32mCL,FrameIndex, MI);
    case X86::ROR8ri:    return MakeMIInst(X86::ROR8mi , FrameIndex, MI);
    case X86::ROR16ri:   return MakeMIInst(X86::ROR16mi, FrameIndex, MI);
    case X86::ROR32ri:   return MakeMIInst(X86::ROR32mi, FrameIndex, MI);
    case X86::ROR8r1:    return MakeMInst(X86::ROR8m1 , FrameIndex, MI);
    case X86::ROR16r1:   return MakeMInst(X86::ROR16m1, FrameIndex, MI);
    case X86::ROR32r1:   return MakeMInst(X86::ROR32m1, FrameIndex, MI);
    case X86::SHLD32rrCL:return MakeMRInst( X86::SHLD32mrCL,FrameIndex, MI);
    case X86::SHLD32rri8:return MakeMRIInst(X86::SHLD32mri8,FrameIndex, MI);
    case X86::SHRD32rrCL:return MakeMRInst( X86::SHRD32mrCL,FrameIndex, MI);
    case X86::SHRD32rri8:return MakeMRIInst(X86::SHRD32mri8,FrameIndex, MI);
    case X86::SHLD16rrCL:return MakeMRInst( X86::SHLD16mrCL,FrameIndex, MI);
    case X86::SHLD16rri8:return MakeMRIInst(X86::SHLD16mri8,FrameIndex, MI);
    case X86::SHRD16rrCL:return MakeMRInst( X86::SHRD16mrCL,FrameIndex, MI);
    case X86::SHRD16rri8:return MakeMRIInst(X86::SHRD16mri8,FrameIndex, MI);
    case X86::SETBr:     return MakeMInst( X86::SETBm,   FrameIndex, MI);
    case X86::SETAEr:    return MakeMInst( X86::SETAEm,  FrameIndex, MI);
    case X86::SETEr:     return MakeMInst( X86::SETEm,   FrameIndex, MI);
    case X86::SETNEr:    return MakeMInst( X86::SETNEm,  FrameIndex, MI);
    case X86::SETBEr:    return MakeMInst( X86::SETBEm,  FrameIndex, MI);
    case X86::SETAr:     return MakeMInst( X86::SETAm,   FrameIndex, MI);
    case X86::SETSr:     return MakeMInst( X86::SETSm,   FrameIndex, MI);
    case X86::SETNSr:    return MakeMInst( X86::SETNSm,  FrameIndex, MI);
    case X86::SETPr:     return MakeMInst( X86::SETPm,   FrameIndex, MI);
    case X86::SETNPr:    return MakeMInst( X86::SETNPm,  FrameIndex, MI);
    case X86::SETLr:     return MakeMInst( X86::SETLm,   FrameIndex, MI);
    case X86::SETGEr:    return MakeMInst( X86::SETGEm,  FrameIndex, MI);
    case X86::SETLEr:    return MakeMInst( X86::SETLEm,  FrameIndex, MI);
    case X86::SETGr:     return MakeMInst( X86::SETGm,   FrameIndex, MI);
    // Alias instructions
    case X86::MOV8r0:    return MakeM0Inst(X86::MOV8mi, FrameIndex, MI);
    case X86::MOV16r0:   return MakeM0Inst(X86::MOV16mi, FrameIndex, MI);
    case X86::MOV32r0:   return MakeM0Inst(X86::MOV32mi, FrameIndex, MI);
    // Alias scalar SSE instructions
    case X86::FsMOVAPSrr: return MakeMRInst(X86::MOVSSmr, FrameIndex, MI);
    case X86::FsMOVAPDrr: return MakeMRInst(X86::MOVSDmr, FrameIndex, MI);
    // Scalar SSE instructions
    case X86::MOVSSrr:   return MakeMRInst(X86::MOVSSmr, FrameIndex, MI);
    case X86::MOVSDrr:   return MakeMRInst(X86::MOVSDmr, FrameIndex, MI);
    // Packed SSE instructions
    case X86::MOVAPSrr:  return MakeMRInst(X86::MOVAPSmr, FrameIndex, MI);
    case X86::MOVAPDrr:  return MakeMRInst(X86::MOVAPDmr, FrameIndex, MI);
    case X86::MOVUPSrr:  return MakeMRInst(X86::MOVUPSmr, FrameIndex, MI);
    case X86::MOVUPDrr:  return MakeMRInst(X86::MOVUPDmr, FrameIndex, MI);
    // Alias packed SSE instructions
    case X86::MOVPS2SSrr:return MakeMRInst(X86::MOVPS2SSmr, FrameIndex, MI);
    case X86::MOVPDI2DIrr:return MakeMRInst(X86::MOVPDI2DImr, FrameIndex, MI);
    }
  } else if (i == 1) {
    switch(MI->getOpcode()) {
    case X86::XCHG8rr:   return MakeRMInst(X86::XCHG8rm ,FrameIndex, MI);
    case X86::XCHG16rr:  return MakeRMInst(X86::XCHG16rm,FrameIndex, MI);
    case X86::XCHG32rr:  return MakeRMInst(X86::XCHG32rm,FrameIndex, MI);
    case X86::MOV8rr:    return MakeRMInst(X86::MOV8rm , FrameIndex, MI);
    case X86::MOV16rr:   return MakeRMInst(X86::MOV16rm, FrameIndex, MI);
    case X86::MOV32rr:   return MakeRMInst(X86::MOV32rm, FrameIndex, MI);
    case X86::CMOVB16rr: return MakeRMInst(X86::CMOVB16rm , FrameIndex, MI);
    case X86::CMOVB32rr: return MakeRMInst(X86::CMOVB32rm , FrameIndex, MI);
    case X86::CMOVAE16rr: return MakeRMInst(X86::CMOVAE16rm , FrameIndex, MI);
    case X86::CMOVAE32rr: return MakeRMInst(X86::CMOVAE32rm , FrameIndex, MI);
    case X86::CMOVE16rr: return MakeRMInst(X86::CMOVE16rm , FrameIndex, MI);
    case X86::CMOVE32rr: return MakeRMInst(X86::CMOVE32rm , FrameIndex, MI);
    case X86::CMOVNE16rr:return MakeRMInst(X86::CMOVNE16rm, FrameIndex, MI);
    case X86::CMOVNE32rr:return MakeRMInst(X86::CMOVNE32rm, FrameIndex, MI);
    case X86::CMOVBE16rr:return MakeRMInst(X86::CMOVBE16rm, FrameIndex, MI);
    case X86::CMOVBE32rr:return MakeRMInst(X86::CMOVBE32rm, FrameIndex, MI);
    case X86::CMOVA16rr:return MakeRMInst(X86::CMOVA16rm, FrameIndex, MI);
    case X86::CMOVA32rr:return MakeRMInst(X86::CMOVA32rm, FrameIndex, MI);
    case X86::CMOVS16rr: return MakeRMInst(X86::CMOVS16rm , FrameIndex, MI);
    case X86::CMOVS32rr: return MakeRMInst(X86::CMOVS32rm , FrameIndex, MI);
    case X86::CMOVNS16rr: return MakeRMInst(X86::CMOVNS16rm , FrameIndex, MI);
    case X86::CMOVNS32rr: return MakeRMInst(X86::CMOVNS32rm , FrameIndex, MI);
    case X86::CMOVP16rr: return MakeRMInst(X86::CMOVP16rm , FrameIndex, MI);
    case X86::CMOVP32rr: return MakeRMInst(X86::CMOVP32rm , FrameIndex, MI);
    case X86::CMOVNP16rr: return MakeRMInst(X86::CMOVNP16rm , FrameIndex, MI);
    case X86::CMOVNP32rr: return MakeRMInst(X86::CMOVNP32rm , FrameIndex, MI);
    case X86::CMOVL16rr: return MakeRMInst(X86::CMOVL16rm , FrameIndex, MI);
    case X86::CMOVL32rr: return MakeRMInst(X86::CMOVL32rm , FrameIndex, MI);
    case X86::CMOVGE16rr: return MakeRMInst(X86::CMOVGE16rm , FrameIndex, MI);
    case X86::CMOVGE32rr: return MakeRMInst(X86::CMOVGE32rm , FrameIndex, MI);
    case X86::CMOVLE16rr: return MakeRMInst(X86::CMOVLE16rm , FrameIndex, MI);
    case X86::CMOVLE32rr: return MakeRMInst(X86::CMOVLE32rm , FrameIndex, MI);
    case X86::CMOVG16rr: return MakeRMInst(X86::CMOVG16rm , FrameIndex, MI);
    case X86::CMOVG32rr: return MakeRMInst(X86::CMOVG32rm , FrameIndex, MI);
    case X86::ADD8rr:    return MakeRMInst(X86::ADD8rm , FrameIndex, MI);
    case X86::ADD16rr:   return MakeRMInst(X86::ADD16rm, FrameIndex, MI);
    case X86::ADD32rr:   return MakeRMInst(X86::ADD32rm, FrameIndex, MI);
    case X86::ADC32rr:   return MakeRMInst(X86::ADC32rm, FrameIndex, MI);
    case X86::SUB8rr:    return MakeRMInst(X86::SUB8rm , FrameIndex, MI);
    case X86::SUB16rr:   return MakeRMInst(X86::SUB16rm, FrameIndex, MI);
    case X86::SUB32rr:   return MakeRMInst(X86::SUB32rm, FrameIndex, MI);
    case X86::SBB32rr:   return MakeRMInst(X86::SBB32rm, FrameIndex, MI);
    case X86::AND8rr:    return MakeRMInst(X86::AND8rm , FrameIndex, MI);
    case X86::AND16rr:   return MakeRMInst(X86::AND16rm, FrameIndex, MI);
    case X86::AND32rr:   return MakeRMInst(X86::AND32rm, FrameIndex, MI);
    case X86::OR8rr:     return MakeRMInst(X86::OR8rm ,  FrameIndex, MI);
    case X86::OR16rr:    return MakeRMInst(X86::OR16rm,  FrameIndex, MI);
    case X86::OR32rr:    return MakeRMInst(X86::OR32rm,  FrameIndex, MI);
    case X86::XOR8rr:    return MakeRMInst(X86::XOR8rm , FrameIndex, MI);
    case X86::XOR16rr:   return MakeRMInst(X86::XOR16rm, FrameIndex, MI);
    case X86::XOR32rr:   return MakeRMInst(X86::XOR32rm, FrameIndex, MI);
    case X86::IMUL16rr:  return MakeRMInst(X86::IMUL16rm,FrameIndex, MI);
    case X86::IMUL32rr:  return MakeRMInst(X86::IMUL32rm,FrameIndex, MI);
    case X86::IMUL16rri: return MakeRMIInst(X86::IMUL16rmi, FrameIndex, MI);
    case X86::IMUL32rri: return MakeRMIInst(X86::IMUL32rmi, FrameIndex, MI);
    case X86::IMUL16rri8:return MakeRMIInst(X86::IMUL16rmi8, FrameIndex, MI);
    case X86::IMUL32rri8:return MakeRMIInst(X86::IMUL32rmi8, FrameIndex, MI);
    case X86::TEST8rr:   return MakeRMInst(X86::TEST8rm ,FrameIndex, MI);
    case X86::TEST16rr:  return MakeRMInst(X86::TEST16rm,FrameIndex, MI);
    case X86::TEST32rr:  return MakeRMInst(X86::TEST32rm,FrameIndex, MI);
    case X86::TEST8ri:   return MakeMIInst(X86::TEST8mi ,FrameIndex, MI);
    case X86::TEST16ri:  return MakeMIInst(X86::TEST16mi,FrameIndex, MI);
    case X86::TEST32ri:  return MakeMIInst(X86::TEST32mi,FrameIndex, MI);
    case X86::CMP8rr:    return MakeRMInst(X86::CMP8rm , FrameIndex, MI);
    case X86::CMP16rr:   return MakeRMInst(X86::CMP16rm, FrameIndex, MI);
    case X86::CMP32rr:   return MakeRMInst(X86::CMP32rm, FrameIndex, MI);
    case X86::CMP8ri:    return MakeRMInst(X86::CMP8mi , FrameIndex, MI);
    case X86::CMP16ri:   return MakeMIInst(X86::CMP16mi, FrameIndex, MI);
    case X86::CMP32ri:   return MakeMIInst(X86::CMP32mi, FrameIndex, MI);
    case X86::CMP16ri8:  return MakeMIInst(X86::CMP16mi8, FrameIndex, MI);
    case X86::CMP32ri8:  return MakeRMInst(X86::CMP32mi8, FrameIndex, MI);
    case X86::MOVSX16rr8:return MakeRMInst(X86::MOVSX16rm8 , FrameIndex, MI);
    case X86::MOVSX32rr8:return MakeRMInst(X86::MOVSX32rm8, FrameIndex, MI);
    case X86::MOVSX32rr16:return MakeRMInst(X86::MOVSX32rm16, FrameIndex, MI);
    case X86::MOVZX16rr8:return MakeRMInst(X86::MOVZX16rm8 , FrameIndex, MI);
    case X86::MOVZX32rr8:return MakeRMInst(X86::MOVZX32rm8, FrameIndex, MI);
    case X86::MOVZX32rr16:return MakeRMInst(X86::MOVZX32rm16, FrameIndex, MI);
    // Alias scalar SSE instructions
    case X86::FsMOVAPSrr:return MakeRMInst(X86::MOVSSrm, FrameIndex, MI);
    case X86::FsMOVAPDrr:return MakeRMInst(X86::MOVSDrm, FrameIndex, MI);
    // Scalar SSE instructions
    case X86::MOVSSrr:   return MakeRMInst(X86::MOVSSrm, FrameIndex, MI);
    case X86::MOVSDrr:   return MakeRMInst(X86::MOVSDrm, FrameIndex, MI);
    case X86::Int_CVTSS2SIrr:
      return MakeRMInst(X86::Int_CVTSS2SIrm, FrameIndex, MI);
    case X86::CVTTSS2SIrr:return MakeRMInst(X86::CVTTSS2SIrm, FrameIndex, MI);
    case X86::Int_CVTSD2SIrr:
      return MakeRMInst(X86::Int_CVTSD2SIrm, FrameIndex, MI);
    case X86::CVTTSD2SIrr:return MakeRMInst(X86::CVTTSD2SIrm, FrameIndex, MI);
    case X86::CVTSS2SDrr:return MakeRMInst(X86::CVTSS2SDrm, FrameIndex, MI);
    case X86::CVTSD2SSrr:return MakeRMInst(X86::CVTSD2SSrm, FrameIndex, MI);
    case X86::CVTSI2SSrr:return MakeRMInst(X86::CVTSI2SSrm, FrameIndex, MI);
    case X86::CVTSI2SDrr:return MakeRMInst(X86::CVTSI2SDrm, FrameIndex, MI);
    case X86::Int_CVTTSS2SIrr:
      return MakeRMInst(X86::Int_CVTTSS2SIrm, FrameIndex, MI);
    case X86::Int_CVTTSD2SIrr:
      return MakeRMInst(X86::Int_CVTTSD2SIrm, FrameIndex, MI);
    case X86::Int_CVTSI2SSrr:
      return MakeRMInst(X86::Int_CVTSI2SSrm, FrameIndex, MI);
    case X86::SQRTSSr:  return MakeRMInst(X86::SQRTSSm, FrameIndex, MI);
    case X86::SQRTSDr:  return MakeRMInst(X86::SQRTSDm, FrameIndex, MI);
    case X86::ADDSSrr:   return MakeRMInst(X86::ADDSSrm, FrameIndex, MI);
    case X86::ADDSDrr:   return MakeRMInst(X86::ADDSDrm, FrameIndex, MI);
    case X86::MULSSrr:   return MakeRMInst(X86::MULSSrm, FrameIndex, MI);
    case X86::MULSDrr:   return MakeRMInst(X86::MULSDrm, FrameIndex, MI);
    case X86::DIVSSrr:   return MakeRMInst(X86::DIVSSrm, FrameIndex, MI);
    case X86::DIVSDrr:   return MakeRMInst(X86::DIVSDrm, FrameIndex, MI);
    case X86::SUBSSrr:   return MakeRMInst(X86::SUBSSrm, FrameIndex, MI);
    case X86::SUBSDrr:   return MakeRMInst(X86::SUBSDrm, FrameIndex, MI);
    case X86::CMPSSrr:   return MakeRMInst(X86::CMPSSrm, FrameIndex, MI);
    case X86::CMPSDrr:   return MakeRMInst(X86::CMPSDrm, FrameIndex, MI);
    case X86::Int_CMPSSrr: return MakeRMInst(X86::Int_CMPSSrm, FrameIndex, MI);
    case X86::Int_CMPSDrr: return MakeRMInst(X86::Int_CMPSDrm, FrameIndex, MI);
    case X86::UCOMISSrr: return MakeRMInst(X86::UCOMISSrm, FrameIndex, MI);
    case X86::UCOMISDrr: return MakeRMInst(X86::UCOMISDrm, FrameIndex, MI);
    case X86::Int_UCOMISSrr:
      return MakeRMInst(X86::Int_UCOMISSrm, FrameIndex, MI);
    case X86::Int_UCOMISDrr:
      return MakeRMInst(X86::Int_UCOMISDrm, FrameIndex, MI);
    case X86::Int_COMISSrr:
      return MakeRMInst(X86::Int_COMISSrm, FrameIndex, MI);
    case X86::Int_COMISDrr:
      return MakeRMInst(X86::Int_COMISDrm, FrameIndex, MI);
    // Packed SSE instructions
    case X86::MOVAPSrr:  return MakeRMInst(X86::MOVAPSrm, FrameIndex, MI);
    case X86::MOVAPDrr:  return MakeRMInst(X86::MOVAPDrm, FrameIndex, MI);
    case X86::MOVUPSrr:  return MakeRMInst(X86::MOVUPSrm, FrameIndex, MI);
    case X86::MOVUPDrr:  return MakeRMInst(X86::MOVUPDrm, FrameIndex, MI);
    case X86::MOVSHDUPrr:return MakeRMInst(X86::MOVSHDUPrm, FrameIndex, MI);
    case X86::MOVSLDUPrr:return MakeRMInst(X86::MOVSLDUPrm, FrameIndex, MI);
    case X86::MOVDDUPrr: return MakeRMInst(X86::MOVDDUPrm, FrameIndex, MI);
    case X86::Int_CVTDQ2PSrr:
      return MakeRMInst(X86::Int_CVTDQ2PSrm, FrameIndex, MI);
    case X86::Int_CVTDQ2PDrr:
      return MakeRMInst(X86::Int_CVTDQ2PDrm, FrameIndex, MI);
    case X86::Int_CVTPS2DQrr:
      return MakeRMInst(X86::Int_CVTPS2DQrm, FrameIndex, MI);
    case X86::Int_CVTTPS2DQrr:
      return MakeRMInst(X86::Int_CVTTPS2DQrm, FrameIndex, MI);
    case X86::Int_CVTPD2DQrr:
      return MakeRMInst(X86::Int_CVTPD2DQrm, FrameIndex, MI);
    case X86::Int_CVTTPD2DQrr:
      return MakeRMInst(X86::Int_CVTTPD2DQrm, FrameIndex, MI);
    case X86::Int_CVTPS2PDrr:
      return MakeRMInst(X86::Int_CVTPS2PDrm, FrameIndex, MI);
    case X86::Int_CVTPD2PSrr:
      return MakeRMInst(X86::Int_CVTPD2PSrm, FrameIndex, MI);
    case X86::Int_CVTSI2SDrr:
      return MakeRMInst(X86::Int_CVTSI2SDrm, FrameIndex, MI);
    case X86::Int_CVTSD2SSrr:
      return MakeRMInst(X86::Int_CVTSD2SSrm, FrameIndex, MI);
    case X86::Int_CVTSS2SDrr:
      return MakeRMInst(X86::Int_CVTSS2SDrm, FrameIndex, MI);
    case X86::ADDPSrr:   return MakeRMInst(X86::ADDPSrm, FrameIndex, MI);
    case X86::ADDPDrr:   return MakeRMInst(X86::ADDPDrm, FrameIndex, MI);
    case X86::SUBPSrr:   return MakeRMInst(X86::SUBPSrm, FrameIndex, MI);
    case X86::SUBPDrr:   return MakeRMInst(X86::SUBPDrm, FrameIndex, MI);
    case X86::MULPSrr:   return MakeRMInst(X86::MULPSrm, FrameIndex, MI);
    case X86::MULPDrr:   return MakeRMInst(X86::MULPDrm, FrameIndex, MI);
    case X86::DIVPSrr:   return MakeRMInst(X86::DIVPSrm, FrameIndex, MI);
    case X86::DIVPDrr:   return MakeRMInst(X86::DIVPDrm, FrameIndex, MI);
    case X86::ADDSUBPSrr:return MakeRMInst(X86::ADDSUBPSrm, FrameIndex, MI);
    case X86::ADDSUBPDrr:return MakeRMInst(X86::ADDSUBPDrm, FrameIndex, MI);
    case X86::HADDPSrr:  return MakeRMInst(X86::HADDPSrm, FrameIndex, MI);
    case X86::HADDPDrr:  return MakeRMInst(X86::HADDPDrm, FrameIndex, MI);
    case X86::HSUBPSrr:  return MakeRMInst(X86::HSUBPSrm, FrameIndex, MI);
    case X86::HSUBPDrr:  return MakeRMInst(X86::HSUBPDrm, FrameIndex, MI);
    case X86::SQRTPSr:   return MakeRMInst(X86::SQRTPSm, FrameIndex, MI);
    case X86::SQRTPDr:   return MakeRMInst(X86::SQRTPDm, FrameIndex, MI);
    case X86::RSQRTPSr:  return MakeRMInst(X86::RSQRTPSm, FrameIndex, MI);
    case X86::RCPPSr:    return MakeRMInst(X86::RCPPSm, FrameIndex, MI);
    case X86::MAXPSrr:   return MakeRMInst(X86::MAXPSrm, FrameIndex, MI);
    case X86::MAXPDrr:   return MakeRMInst(X86::MAXPDrm, FrameIndex, MI);
    case X86::MINPSrr:   return MakeRMInst(X86::MINPSrm, FrameIndex, MI);
    case X86::MINPDrr:   return MakeRMInst(X86::MINPDrm, FrameIndex, MI);
    case X86::ANDPSrr:   return MakeRMInst(X86::ANDPSrm, FrameIndex, MI);
    case X86::ANDPDrr:   return MakeRMInst(X86::ANDPDrm, FrameIndex, MI);
    case X86::ORPSrr:    return MakeRMInst(X86::ORPSrm, FrameIndex, MI);
    case X86::ORPDrr:    return MakeRMInst(X86::ORPDrm, FrameIndex, MI);
    case X86::XORPSrr:   return MakeRMInst(X86::XORPSrm, FrameIndex, MI);
    case X86::XORPDrr:   return MakeRMInst(X86::XORPDrm, FrameIndex, MI);
    case X86::ANDNPSrr:  return MakeRMInst(X86::ANDNPSrm, FrameIndex, MI);
    case X86::ANDNPDrr:  return MakeRMInst(X86::ANDNPDrm, FrameIndex, MI);
    case X86::CMPPSrri:  return MakeRMIInst(X86::CMPPSrmi, FrameIndex, MI);
    case X86::CMPPDrri:  return MakeRMIInst(X86::CMPPDrmi, FrameIndex, MI);
    case X86::SHUFPSrri: return MakeRMIInst(X86::SHUFPSrmi, FrameIndex, MI);
    case X86::SHUFPDrri: return MakeRMIInst(X86::SHUFPDrmi, FrameIndex, MI);
    case X86::UNPCKHPSrr:return MakeRMInst(X86::UNPCKHPSrm, FrameIndex, MI);
    case X86::UNPCKHPDrr:return MakeRMInst(X86::UNPCKHPDrm, FrameIndex, MI);
    case X86::UNPCKLPSrr:return MakeRMInst(X86::UNPCKLPSrm, FrameIndex, MI);
    case X86::UNPCKLPDrr:return MakeRMInst(X86::UNPCKLPDrm, FrameIndex, MI);
    case X86::PADDBrr:   return MakeRMInst(X86::PADDBrm, FrameIndex, MI);
    case X86::PADDWrr:   return MakeRMInst(X86::PADDWrm, FrameIndex, MI);
    case X86::PADDDrr:   return MakeRMInst(X86::PADDDrm, FrameIndex, MI);
    case X86::PADDSBrr:  return MakeRMInst(X86::PADDSBrm, FrameIndex, MI);
    case X86::PADDSWrr:  return MakeRMInst(X86::PADDSWrm, FrameIndex, MI);
    case X86::PSUBBrr:   return MakeRMInst(X86::PSUBBrm, FrameIndex, MI);
    case X86::PSUBWrr:   return MakeRMInst(X86::PSUBWrm, FrameIndex, MI);
    case X86::PSUBDrr:   return MakeRMInst(X86::PSUBDrm, FrameIndex, MI);
    case X86::PSUBSBrr:  return MakeRMInst(X86::PSUBSBrm, FrameIndex, MI);
    case X86::PSUBSWrr:  return MakeRMInst(X86::PSUBSWrm, FrameIndex, MI);
    case X86::PMULHUWrr: return MakeRMInst(X86::PMULHUWrm, FrameIndex, MI);
    case X86::PMULHWrr:  return MakeRMInst(X86::PMULHWrm, FrameIndex, MI);
    case X86::PMULLWrr:  return MakeRMInst(X86::PMULLWrm, FrameIndex, MI);
    case X86::PMULUDQrr: return MakeRMInst(X86::PMULUDQrm, FrameIndex, MI);
    case X86::PMADDWDrr: return MakeRMInst(X86::PMADDWDrm, FrameIndex, MI);
    case X86::PAVGBrr:   return MakeRMInst(X86::PAVGBrm, FrameIndex, MI);
    case X86::PAVGWrr:   return MakeRMInst(X86::PAVGWrm, FrameIndex, MI);
    case X86::PMAXUBrr:  return MakeRMInst(X86::PMAXUBrm, FrameIndex, MI);
    case X86::PMAXSWrr:  return MakeRMInst(X86::PMAXSWrm, FrameIndex, MI);
    case X86::PMINUBrr:  return MakeRMInst(X86::PMINUBrm, FrameIndex, MI);
    case X86::PMINSWrr:  return MakeRMInst(X86::PMINSWrm, FrameIndex, MI);
    case X86::PSADBWrr:  return MakeRMInst(X86::PSADBWrm, FrameIndex, MI);
    case X86::PSLLWrr:   return MakeRMInst(X86::PSLLWrm, FrameIndex, MI);
    case X86::PSLLDrr:   return MakeRMInst(X86::PSLLDrm, FrameIndex, MI);
    case X86::PSLLQrr:   return MakeRMInst(X86::PSLLQrm, FrameIndex, MI);
    case X86::PSRLWrr:   return MakeRMInst(X86::PSRLWrm, FrameIndex, MI);
    case X86::PSRLDrr:   return MakeRMInst(X86::PSRLDrm, FrameIndex, MI);
    case X86::PSRLQrr:   return MakeRMInst(X86::PSRLQrm, FrameIndex, MI);
    case X86::PSRAWrr:   return MakeRMInst(X86::PSRAWrm, FrameIndex, MI);
    case X86::PSRADrr:   return MakeRMInst(X86::PSRADrm, FrameIndex, MI);
    case X86::PANDrr:    return MakeRMInst(X86::PANDrm, FrameIndex, MI);
    case X86::PORrr:     return MakeRMInst(X86::PORrm, FrameIndex, MI);
    case X86::PXORrr:    return MakeRMInst(X86::PXORrm, FrameIndex, MI);
    case X86::PANDNrr:   return MakeRMInst(X86::PANDNrm, FrameIndex, MI);
    case X86::PCMPEQBrr: return MakeRMInst(X86::PCMPEQBrm, FrameIndex, MI);
    case X86::PCMPEQWrr: return MakeRMInst(X86::PCMPEQWrm, FrameIndex, MI);
    case X86::PCMPEQDrr: return MakeRMInst(X86::PCMPEQDrm, FrameIndex, MI);
    case X86::PCMPGTBrr: return MakeRMInst(X86::PCMPGTBrm, FrameIndex, MI);
    case X86::PCMPGTWrr: return MakeRMInst(X86::PCMPGTWrm, FrameIndex, MI);
    case X86::PCMPGTDrr: return MakeRMInst(X86::PCMPGTDrm, FrameIndex, MI);
    case X86::PACKSSWBrr:return MakeRMInst(X86::PACKSSWBrm, FrameIndex, MI);
    case X86::PACKSSDWrr:return MakeRMInst(X86::PACKSSDWrm, FrameIndex, MI);
    case X86::PACKUSWBrr:return MakeRMInst(X86::PACKUSWBrm, FrameIndex, MI);
    case X86::PSHUFDri:  return MakeRMIInst(X86::PSHUFDmi, FrameIndex, MI);
    case X86::PSHUFHWri: return MakeRMIInst(X86::PSHUFHWmi, FrameIndex, MI);
    case X86::PSHUFLWri: return MakeRMIInst(X86::PSHUFLWmi, FrameIndex, MI);
    case X86::PUNPCKLBWrr:return MakeRMInst(X86::PUNPCKLBWrm, FrameIndex, MI);
    case X86::PUNPCKLWDrr:return MakeRMInst(X86::PUNPCKLWDrm, FrameIndex, MI);
    case X86::PUNPCKLDQrr:return MakeRMInst(X86::PUNPCKLDQrm, FrameIndex, MI);
    case X86::PUNPCKLQDQrr:return MakeRMInst(X86::PUNPCKLQDQrm, FrameIndex, MI);
    case X86::PUNPCKHBWrr:return MakeRMInst(X86::PUNPCKHBWrm, FrameIndex, MI);
    case X86::PUNPCKHWDrr:return MakeRMInst(X86::PUNPCKHWDrm, FrameIndex, MI);
    case X86::PUNPCKHDQrr:return MakeRMInst(X86::PUNPCKHDQrm, FrameIndex, MI);
    case X86::PUNPCKHQDQrr:return MakeRMInst(X86::PUNPCKHQDQrm, FrameIndex, MI);
    case X86::PINSRWrri:  return MakeRMIInst(X86::PINSRWrmi, FrameIndex, MI);
    // Alias packed SSE instructions
    case X86::MOVSS2PSrr:return MakeRMInst(X86::MOVSS2PSrm, FrameIndex, MI);
    case X86::MOVSD2PDrr:return MakeRMInst(X86::MOVSD2PDrm, FrameIndex, MI);
    case X86::MOVDI2PDIrr:return MakeRMInst(X86::MOVDI2PDIrm, FrameIndex, MI);
    case X86::MOVQI2PQIrr:return MakeRMInst(X86::MOVQI2PQIrm, FrameIndex, MI);
    }
  }
  if (PrintFailedFusing)
    std::cerr << "We failed to fuse ("
              << ((i == 1) ? "r" : "s") << "): " << *MI;
  return NULL;
}

const unsigned *X86RegisterInfo::getCalleeSaveRegs() const {
  static const unsigned CalleeSaveRegs[] = {
    X86::ESI, X86::EDI, X86::EBX, X86::EBP,  0
  };
  return CalleeSaveRegs;
}

const TargetRegisterClass* const*
X86RegisterInfo::getCalleeSaveRegClasses() const {
  static const TargetRegisterClass * const CalleeSaveRegClasses[] = {
    &X86::GR32RegClass, &X86::GR32RegClass,
    &X86::GR32RegClass, &X86::GR32RegClass,  0
  };
  return CalleeSaveRegClasses;
}

//===----------------------------------------------------------------------===//
// Stack Frame Processing methods
//===----------------------------------------------------------------------===//

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
//
static bool hasFP(MachineFunction &MF) {
  return (NoFramePointerElim || 
          MF.getFrameInfo()->hasVarSizedObjects() ||
          MF.getInfo<X86FunctionInfo>()->getForceFramePointer());
}

void X86RegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (hasFP(MF)) {
    // If we have a frame pointer, turn the adjcallstackup instruction into a
    // 'sub ESP, <amt>' and the adjcallstackdown instruction into 'add ESP,
    // <amt>'
    MachineInstr *Old = I;
    unsigned Amount = Old->getOperand(0).getImmedValue();
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
      Amount = (Amount+Align-1)/Align*Align;

      MachineInstr *New = 0;
      if (Old->getOpcode() == X86::ADJCALLSTACKDOWN) {
        New=BuildMI(X86::SUB32ri, 1, X86::ESP, MachineOperand::UseAndDef)
              .addImm(Amount);
      } else {
        assert(Old->getOpcode() == X86::ADJCALLSTACKUP);
        // factor out the amount the callee already popped.
        unsigned CalleeAmt = Old->getOperand(1).getImmedValue();
        Amount -= CalleeAmt;
        if (Amount) {
          unsigned Opc = Amount < 128 ? X86::ADD32ri8 : X86::ADD32ri;
          New = BuildMI(Opc, 1, X86::ESP,
                        MachineOperand::UseAndDef).addImm(Amount);
        }
      }

      // Replace the pseudo instruction with a new instruction...
      if (New) MBB.insert(I, New);
    }
  } else if (I->getOpcode() == X86::ADJCALLSTACKUP) {
    // If we are performing frame pointer elimination and if the callee pops
    // something off the stack pointer, add it back.  We do this until we have
    // more advanced stack pointer tracking ability.
    if (unsigned CalleeAmt = I->getOperand(1).getImmedValue()) {
      unsigned Opc = CalleeAmt < 128 ? X86::SUB32ri8 : X86::SUB32ri;
      MachineInstr *New =
        BuildMI(Opc, 1, X86::ESP,
                MachineOperand::UseAndDef).addImm(CalleeAmt);
      MBB.insert(I, New);
    }
  }

  MBB.erase(I);
}

void X86RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II) const{
  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  while (!MI.getOperand(i).isFrameIndex()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getFrameIndex();

  // This must be part of a four operand memory reference.  Replace the
  // FrameIndex with base register with EBP.  Add add an offset to the offset.
  MI.getOperand(i).ChangeToRegister(hasFP(MF) ? X86::EBP : X86::ESP);

  // Now add the frame object offset to the offset from EBP.
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) +
               MI.getOperand(i+3).getImmedValue()+4;

  if (!hasFP(MF))
    Offset += MF.getFrameInfo()->getStackSize();
  else
    Offset += 4;  // Skip the saved EBP

  MI.getOperand(i+3).ChangeToImmediate(Offset);
}

void
X86RegisterInfo::processFunctionBeforeFrameFinalized(MachineFunction &MF) const{
  if (hasFP(MF)) {
    // Create a frame entry for the EBP register that must be saved.
    int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, -8);
    assert(FrameIdx == MF.getFrameInfo()->getObjectIndexBegin() &&
           "Slot for EBP register must be last in order to be found!");
  }
}

void X86RegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
  const Function* Fn = MF.getFunction();
  const X86Subtarget* Subtarget = &MF.getTarget().getSubtarget<X86Subtarget>();
  MachineInstr *MI;
  
  // Get the number of bytes to allocate from the FrameInfo
  unsigned NumBytes = MFI->getStackSize();
  if (MFI->hasCalls() || MF.getFrameInfo()->hasVarSizedObjects()) {
    // When we have no frame pointer, we reserve argument space for call sites
    // in the function immediately on entry to the current function.  This
    // eliminates the need for add/sub ESP brackets around call sites.
    //
    if (!hasFP(MF))
      NumBytes += MFI->getMaxCallFrameSize();

    // Round the size to a multiple of the alignment (don't forget the 4 byte
    // offset though).
    NumBytes = ((NumBytes+4)+Align-1)/Align*Align - 4;
  }

  // Update frame info to pretend that this is part of the stack...
  MFI->setStackSize(NumBytes);

  if (NumBytes) {   // adjust stack pointer: ESP -= numbytes
    if (NumBytes >= 4096 && Subtarget->TargetType == X86Subtarget::isCygwin) {
      // Function prologue calls _alloca to probe the stack when allocating  
      // more than 4k bytes in one go. Touching the stack at 4K increments is  
      // necessary to ensure that the guard pages used by the OS virtual memory
      // manager are allocated in correct sequence.
      MI = BuildMI(X86::MOV32ri, 2, X86::EAX).addImm(NumBytes);
      MBB.insert(MBBI, MI);
      MI = BuildMI(X86::CALLpcrel32, 1).addExternalSymbol("_alloca");
      MBB.insert(MBBI, MI);
    } else {
      unsigned Opc = NumBytes < 128 ? X86::SUB32ri8 : X86::SUB32ri;
      MI = BuildMI(Opc, 1, X86::ESP,MachineOperand::UseAndDef).addImm(NumBytes);
      MBB.insert(MBBI, MI);
    }
  }

  if (hasFP(MF)) {
    // Get the offset of the stack slot for the EBP register... which is
    // guaranteed to be the last slot by processFunctionBeforeFrameFinalized.
    int EBPOffset = MFI->getObjectOffset(MFI->getObjectIndexBegin())+4;

    // Save EBP into the appropriate stack slot...
    MI = addRegOffset(BuildMI(X86::MOV32mr, 5),    // mov [ESP-<offset>], EBP
                      X86::ESP, EBPOffset+NumBytes).addReg(X86::EBP);
    MBB.insert(MBBI, MI);

    // Update EBP with the new base value...
    if (NumBytes == 4)    // mov EBP, ESP
      MI = BuildMI(X86::MOV32rr, 2, X86::EBP).addReg(X86::ESP);
    else                  // lea EBP, [ESP+StackSize]
      MI = addRegOffset(BuildMI(X86::LEA32r, 5, X86::EBP), X86::ESP,NumBytes-4);

    MBB.insert(MBBI, MI);
  }

  // If it's main() on Cygwin\Mingw32 we should align stack as well
  if (Fn->hasExternalLinkage() && Fn->getName() == "main" &&
      Subtarget->TargetType == X86Subtarget::isCygwin) {
    MI = BuildMI(X86::AND32ri, 2, X86::ESP).addImm(-Align);
    MBB.insert(MBBI, MI);

    // Probe the stack
    MI = BuildMI(X86::MOV32ri, 2, X86::EAX).addImm(Align);
    MBB.insert(MBBI, MI);
    MI = BuildMI(X86::CALLpcrel32, 1).addExternalSymbol("_alloca");
    MBB.insert(MBBI, MI);
  }
}

void X86RegisterInfo::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineBasicBlock::iterator MBBI = prior(MBB.end());

  switch (MBBI->getOpcode()) {
  case X86::RET:
  case X86::RETI:
  case X86::TAILJMPd:
  case X86::TAILJMPr:
  case X86::TAILJMPm: break;  // These are ok
  default:
    assert(0 && "Can only insert epilog into returning blocks");
  }

  if (hasFP(MF)) {
    // Get the offset of the stack slot for the EBP register... which is
    // guaranteed to be the last slot by processFunctionBeforeFrameFinalized.
    int EBPOffset = MFI->getObjectOffset(MFI->getObjectIndexEnd()-1)+4;

    // mov ESP, EBP
    BuildMI(MBB, MBBI, X86::MOV32rr, 1,X86::ESP).addReg(X86::EBP);

    // pop EBP
    BuildMI(MBB, MBBI, X86::POP32r, 0, X86::EBP);
  } else {
    // Get the number of bytes allocated from the FrameInfo...
    unsigned NumBytes = MFI->getStackSize();

    if (NumBytes) {    // adjust stack pointer back: ESP += numbytes
      // If there is an ADD32ri or SUB32ri of ESP immediately before this
      // instruction, merge the two instructions.
      if (MBBI != MBB.begin()) {
        MachineBasicBlock::iterator PI = prior(MBBI);
        if ((PI->getOpcode() == X86::ADD32ri || 
             PI->getOpcode() == X86::ADD32ri8) &&
            PI->getOperand(0).getReg() == X86::ESP) {
          NumBytes += PI->getOperand(1).getImmedValue();
          MBB.erase(PI);
        } else if ((PI->getOpcode() == X86::SUB32ri ||
                    PI->getOpcode() == X86::SUB32ri8) &&
                   PI->getOperand(0).getReg() == X86::ESP) {
          NumBytes -= PI->getOperand(1).getImmedValue();
          MBB.erase(PI);
        } else if (PI->getOpcode() == X86::ADJSTACKPTRri) {
          NumBytes += PI->getOperand(1).getImmedValue();
          MBB.erase(PI);
        }
      }

      if (NumBytes > 0) {
        unsigned Opc = NumBytes < 128 ? X86::ADD32ri8 : X86::ADD32ri;
        BuildMI(MBB, MBBI, Opc, 2)
          .addReg(X86::ESP, MachineOperand::UseAndDef).addImm(NumBytes);
      } else if ((int)NumBytes < 0) {
        unsigned Opc = -NumBytes < 128 ? X86::SUB32ri8 : X86::SUB32ri;
        BuildMI(MBB, MBBI, Opc, 2)
          .addReg(X86::ESP, MachineOperand::UseAndDef).addImm(-NumBytes);
      }
    }
  }
}

unsigned X86RegisterInfo::getRARegister() const {
  return X86::ST0;  // use a non-register register
}

unsigned X86RegisterInfo::getFrameRegister(MachineFunction &MF) const {
  return hasFP(MF) ? X86::EBP : X86::ESP;
}

namespace llvm {
unsigned getX86SubSuperRegister(unsigned Reg, MVT::ValueType VT, bool High) {
  switch (VT) {
  default: return Reg;
  case MVT::i8:
    if (High) {
      switch (Reg) {
      default: return Reg;
      case X86::AH: case X86::AL: case X86::AX: case X86::EAX:
        return X86::AH;
      case X86::DH: case X86::DL: case X86::DX: case X86::EDX:
        return X86::DH;
      case X86::CH: case X86::CL: case X86::CX: case X86::ECX:
        return X86::CH;
      case X86::BH: case X86::BL: case X86::BX: case X86::EBX:
        return X86::BH;
      }
    } else {
      switch (Reg) {
      default: return Reg;
      case X86::AH: case X86::AL: case X86::AX: case X86::EAX:
        return X86::AL;
      case X86::DH: case X86::DL: case X86::DX: case X86::EDX:
        return X86::DL;
      case X86::CH: case X86::CL: case X86::CX: case X86::ECX:
        return X86::CL;
      case X86::BH: case X86::BL: case X86::BX: case X86::EBX:
        return X86::BL;
      }
    }
  case MVT::i16:
    switch (Reg) {
    default: return Reg;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX:
      return X86::AX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX:
      return X86::DX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX:
      return X86::CX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX:
      return X86::BX;
    case X86::ESI:
      return X86::SI;
    case X86::EDI:
      return X86::DI;
    case X86::EBP:
      return X86::BP;
    case X86::ESP:
      return X86::SP;
    }
  case MVT::i32:
    switch (Reg) {
    default: return true;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX:
      return X86::EAX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX:
      return X86::EDX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX:
      return X86::ECX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX:
      return X86::EBX;
    case X86::SI:
      return X86::ESI;
    case X86::DI:
      return X86::EDI;
    case X86::BP:
      return X86::EBP;
    case X86::SP:
      return X86::ESP;
    }
  }

  return Reg;
}
}

#include "X86GenRegisterInfo.inc"

