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
#include "X86InstrBuilder.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "Support/CommandLine.h"
#include "Support/STLExtras.h"
using namespace llvm;

namespace {
  cl::opt<bool>
  NoFPElim("disable-fp-elim",
	   cl::desc("Disable frame pointer elimination optimization"));
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

static unsigned getIdx(const TargetRegisterClass *RC) {
  switch (RC->getSize()) {
  default: assert(0 && "Invalid data size!");
  case 1:  return 0;
  case 2:  return 1;
  case 4:  return 2;
  case 10: return 3;
  }
}

int X86RegisterInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                         unsigned SrcReg, int FrameIdx,
                                         const TargetRegisterClass *RC) const {
  static const unsigned Opcode[] =
    { X86::MOV8mr, X86::MOV16mr, X86::MOV32mr, X86::FSTP80m };
  MachineInstr *I = addFrameReference(BuildMI(Opcode[getIdx(RC)], 5),
				       FrameIdx).addReg(SrcReg);
  MBB.insert(MI, I);
  return 1;
}

int X86RegisterInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MI,
                                          unsigned DestReg, int FrameIdx,
                                          const TargetRegisterClass *RC) const{
  static const unsigned Opcode[] =
    { X86::MOV8rm, X86::MOV16rm, X86::MOV32rm, X86::FLD80m };
  unsigned OC = Opcode[getIdx(RC)];
  MBB.insert(MI, addFrameReference(BuildMI(OC, 4, DestReg), FrameIdx));
  return 1;
}

int X86RegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MI,
                                  unsigned DestReg, unsigned SrcReg,
                                  const TargetRegisterClass *RC) const {
  static const unsigned Opcode[] =
    { X86::MOV8rr, X86::MOV16rr, X86::MOV32rr, X86::FpMOV };
  MBB.insert(MI, BuildMI(Opcode[getIdx(RC)],1,DestReg).addReg(SrcReg));
  return 1;
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
  return addFrameReference(BuildMI(Opcode, 5), FrameIndex)
      .addReg(MI->getOperand(1).getReg())
      .addZImm(MI->getOperand(2).getImmedValue());
}

static MachineInstr *MakeMIInst(unsigned Opcode, unsigned FrameIndex,
                                MachineInstr *MI) {
  if (MI->getOperand(1).isImmediate())
    return addFrameReference(BuildMI(Opcode, 5), FrameIndex)
      .addZImm(MI->getOperand(1).getImmedValue());
  else if (MI->getOperand(1).isGlobalAddress())
    return addFrameReference(BuildMI(Opcode, 5), FrameIndex)
      .addGlobalAddress(MI->getOperand(1).getGlobal());
  assert(0 && "Unknown operand for MakeMI!");
  return 0;
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
  return addFrameReference(BuildMI(Opcode, 5, op.getReg(), op.getUseType()),
                        FrameIndex).addZImm(MI->getOperand(2).getImmedValue());
}


MachineInstr* X86RegisterInfo::foldMemoryOperand(MachineInstr* MI,
                                                 unsigned i,
                                                 int FrameIndex) const {
  if (NoFusing) return NULL;

  /// FIXME: This should obviously be autogenerated by tablegen when patterns
  /// are available!
  MachineBasicBlock& MBB = *MI->getParent();
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
    case X86::ADC32rr:   return MakeMRInst(X86::ADC32mr, FrameIndex, MI);
    case X86::ADC32ri:   return MakeMIInst(X86::ADC32mi, FrameIndex, MI);
    case X86::ADD8ri:    return MakeMIInst(X86::ADD8mi , FrameIndex, MI);
    case X86::ADD16ri:   return MakeMIInst(X86::ADD16mi, FrameIndex, MI);
    case X86::ADD32ri:   return MakeMIInst(X86::ADD32mi, FrameIndex, MI);
    case X86::SUB8rr:    return MakeMRInst(X86::SUB8mr , FrameIndex, MI);
    case X86::SUB16rr:   return MakeMRInst(X86::SUB16mr, FrameIndex, MI);
    case X86::SUB32rr:   return MakeMRInst(X86::SUB32mr, FrameIndex, MI);
    case X86::SBB32rr:   return MakeMRInst(X86::SBB32mr, FrameIndex, MI);
    case X86::SBB32ri:   return MakeMIInst(X86::SBB32mi, FrameIndex, MI);
    case X86::SUB8ri:    return MakeMIInst(X86::SUB8mi , FrameIndex, MI);
    case X86::SUB16ri:   return MakeMIInst(X86::SUB16mi, FrameIndex, MI);
    case X86::SUB32ri:   return MakeMIInst(X86::SUB32mi, FrameIndex, MI);
    case X86::AND8rr:    return MakeMRInst(X86::AND8mr , FrameIndex, MI);
    case X86::AND16rr:   return MakeMRInst(X86::AND16mr, FrameIndex, MI);
    case X86::AND32rr:   return MakeMRInst(X86::AND32mr, FrameIndex, MI);
    case X86::AND8ri:    return MakeMIInst(X86::AND8mi , FrameIndex, MI);
    case X86::AND16ri:   return MakeMIInst(X86::AND16mi, FrameIndex, MI);
    case X86::AND32ri:   return MakeMIInst(X86::AND32mi, FrameIndex, MI);
    case X86::OR8rr:     return MakeMRInst(X86::OR8mr ,  FrameIndex, MI);
    case X86::OR16rr:    return MakeMRInst(X86::OR16mr,  FrameIndex, MI);
    case X86::OR32rr:    return MakeMRInst(X86::OR32mr,  FrameIndex, MI);
    case X86::OR8ri:     return MakeMIInst(X86::OR8mi ,  FrameIndex, MI);
    case X86::OR16ri:    return MakeMIInst(X86::OR16mi,  FrameIndex, MI);
    case X86::OR32ri:    return MakeMIInst(X86::OR32mi,  FrameIndex, MI);
    case X86::XOR8rr:    return MakeMRInst(X86::XOR8mr , FrameIndex, MI);
    case X86::XOR16rr:   return MakeMRInst(X86::XOR16mr, FrameIndex, MI);
    case X86::XOR32rr:   return MakeMRInst(X86::XOR32mr, FrameIndex, MI);
    case X86::XOR8ri:    return MakeMIInst(X86::XOR8mi , FrameIndex, MI);
    case X86::XOR16ri:   return MakeMIInst(X86::XOR16mi, FrameIndex, MI);
    case X86::XOR32ri:   return MakeMIInst(X86::XOR32mi, FrameIndex, MI);
    case X86::SHL8rCL:   return MakeMInst( X86::SHL8mCL ,FrameIndex, MI);
    case X86::SHL16rCL:  return MakeMInst( X86::SHL16mCL,FrameIndex, MI);
    case X86::SHL32rCL:  return MakeMInst( X86::SHL32mCL,FrameIndex, MI);
    case X86::SHL8ri:    return MakeMIInst(X86::SHL8mi , FrameIndex, MI);
    case X86::SHL16ri:   return MakeMIInst(X86::SHL16mi, FrameIndex, MI);
    case X86::SHL32ri:   return MakeMIInst(X86::SHL32mi, FrameIndex, MI);
    case X86::SHR8rCL:   return MakeMInst( X86::SHR8mCL ,FrameIndex, MI);
    case X86::SHR16rCL:  return MakeMInst( X86::SHR16mCL,FrameIndex, MI);
    case X86::SHR32rCL:  return MakeMInst( X86::SHR32mCL,FrameIndex, MI);
    case X86::SHR8ri:    return MakeMIInst(X86::SHR8mi , FrameIndex, MI);
    case X86::SHR16ri:   return MakeMIInst(X86::SHR16mi, FrameIndex, MI);
    case X86::SHR32ri:   return MakeMIInst(X86::SHR32mi, FrameIndex, MI);
    case X86::SAR8rCL:   return MakeMInst( X86::SAR8mCL ,FrameIndex, MI);
    case X86::SAR16rCL:  return MakeMInst( X86::SAR16mCL,FrameIndex, MI);
    case X86::SAR32rCL:  return MakeMInst( X86::SAR32mCL,FrameIndex, MI);
    case X86::SAR8ri:    return MakeMIInst(X86::SAR8mi , FrameIndex, MI);
    case X86::SAR16ri:   return MakeMIInst(X86::SAR16mi, FrameIndex, MI);
    case X86::SAR32ri:   return MakeMIInst(X86::SAR32mi, FrameIndex, MI);
    case X86::SHLD32rrCL:return MakeMRInst( X86::SHLD32mrCL,FrameIndex, MI);
    case X86::SHLD32rri8:return MakeMRIInst(X86::SHLD32mri8,FrameIndex, MI);
    case X86::SHRD32rrCL:return MakeMRInst( X86::SHRD32mrCL,FrameIndex, MI);
    case X86::SHRD32rri8:return MakeMRIInst(X86::SHRD32mri8,FrameIndex, MI);
    case X86::SETBr:     return MakeMInst( X86::SETBm,   FrameIndex, MI);
    case X86::SETAEr:    return MakeMInst( X86::SETAEm,  FrameIndex, MI);
    case X86::SETEr:     return MakeMInst( X86::SETEm,   FrameIndex, MI);
    case X86::SETNEr:    return MakeMInst( X86::SETNEm,  FrameIndex, MI);
    case X86::SETBEr:    return MakeMInst( X86::SETBEm,  FrameIndex, MI);
    case X86::SETAr:     return MakeMInst( X86::SETAm,   FrameIndex, MI);
    case X86::SETSr:     return MakeMInst( X86::SETSm,   FrameIndex, MI);
    case X86::SETNSr:    return MakeMInst( X86::SETNSm,  FrameIndex, MI);
    case X86::SETLr:     return MakeMInst( X86::SETLm,   FrameIndex, MI);
    case X86::SETGEr:    return MakeMInst( X86::SETGEm,  FrameIndex, MI);
    case X86::SETLEr:    return MakeMInst( X86::SETLEm,  FrameIndex, MI);
    case X86::SETGr:     return MakeMInst( X86::SETGm,   FrameIndex, MI);
    case X86::TEST8rr:   return MakeMRInst(X86::TEST8mr ,FrameIndex, MI);
    case X86::TEST16rr:  return MakeMRInst(X86::TEST16mr,FrameIndex, MI);
    case X86::TEST32rr:  return MakeMRInst(X86::TEST32mr,FrameIndex, MI);
    case X86::TEST8ri:   return MakeMIInst(X86::TEST8mi ,FrameIndex, MI);
    case X86::TEST16ri:  return MakeMIInst(X86::TEST16mi,FrameIndex, MI);
    case X86::TEST32ri:  return MakeMIInst(X86::TEST32mi,FrameIndex, MI);
    case X86::CMP8rr:    return MakeMRInst(X86::CMP8mr , FrameIndex, MI);
    case X86::CMP16rr:   return MakeMRInst(X86::CMP16mr, FrameIndex, MI);
    case X86::CMP32rr:   return MakeMRInst(X86::CMP32mr, FrameIndex, MI);
    case X86::CMP8ri:    return MakeMIInst(X86::CMP8mi , FrameIndex, MI);
    case X86::CMP16ri:   return MakeMIInst(X86::CMP16mi, FrameIndex, MI);
    case X86::CMP32ri:   return MakeMIInst(X86::CMP32mi, FrameIndex, MI);
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
    case X86::TEST8rr:   return MakeRMInst(X86::TEST8rm ,FrameIndex, MI);
    case X86::TEST16rr:  return MakeRMInst(X86::TEST16rm,FrameIndex, MI);
    case X86::TEST32rr:  return MakeRMInst(X86::TEST32rm,FrameIndex, MI);
    case X86::IMUL16rr:  return MakeRMInst(X86::IMUL16rm,FrameIndex, MI);
    case X86::IMUL32rr:  return MakeRMInst(X86::IMUL32rm,FrameIndex, MI);
    case X86::IMUL16rri: return MakeRMIInst(X86::IMUL16rmi, FrameIndex, MI);
    case X86::IMUL32rri: return MakeRMIInst(X86::IMUL32rmi, FrameIndex, MI);
    case X86::CMP8rr:    return MakeRMInst(X86::CMP8rm , FrameIndex, MI);
    case X86::CMP16rr:   return MakeRMInst(X86::CMP16rm, FrameIndex, MI);
    case X86::CMP32rr:   return MakeRMInst(X86::CMP32rm, FrameIndex, MI);
    case X86::MOVSX16rr8:return MakeRMInst(X86::MOVSX16rm8 , FrameIndex, MI);
    case X86::MOVSX32rr8:return MakeRMInst(X86::MOVSX32rm8, FrameIndex, MI);
    case X86::MOVSX32rr16:return MakeRMInst(X86::MOVSX32rm16, FrameIndex, MI);
    case X86::MOVZX16rr8:return MakeRMInst(X86::MOVZX16rm8 , FrameIndex, MI);
    case X86::MOVZX32rr8:  return MakeRMInst(X86::MOVZX32rm8, FrameIndex, MI);
    case X86::MOVZX32rr16:return MakeRMInst(X86::MOVZX32rm16, FrameIndex, MI);
    }
  }
  if (PrintFailedFusing)
    std::cerr << "We failed to fuse: " << *MI;
  return NULL;
}

//===----------------------------------------------------------------------===//
// Stack Frame Processing methods
//===----------------------------------------------------------------------===//

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
//
static bool hasFP(MachineFunction &MF) {
  return NoFPElim || MF.getFrameInfo()->hasVarSizedObjects();
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

      MachineInstr *New;
      if (Old->getOpcode() == X86::ADJCALLSTACKDOWN) {
	New=BuildMI(X86::SUB32ri, 1, X86::ESP, MachineOperand::UseAndDef)
              .addZImm(Amount);
      } else {
	assert(Old->getOpcode() == X86::ADJCALLSTACKUP);
	New=BuildMI(X86::ADD32ri, 1, X86::ESP, MachineOperand::UseAndDef)
              .addZImm(Amount);
      }

      // Replace the pseudo instruction with a new instruction...
      MBB.insert(I, New);
    }
  }

  MBB.erase(I);
}

void X86RegisterInfo::eliminateFrameIndex(MachineFunction &MF,
                                         MachineBasicBlock::iterator II) const {
  unsigned i = 0;
  MachineInstr &MI = *II;
  while (!MI.getOperand(i).isFrameIndex()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getFrameIndex();

  // This must be part of a four operand memory reference.  Replace the
  // FrameIndex with base register with EBP.  Add add an offset to the offset.
  MI.SetMachineOperandReg(i, hasFP(MF) ? X86::EBP : X86::ESP);

  // Now add the frame object offset to the offset from EBP.
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) +
               MI.getOperand(i+3).getImmedValue()+4;

  if (!hasFP(MF))
    Offset += MF.getFrameInfo()->getStackSize();
  else
    Offset += 4;  // Skip the saved EBP

  MI.SetMachineOperandConst(i+3, MachineOperand::MO_SignExtendedImmed, Offset);
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
  MachineInstr *MI;

  // Get the number of bytes to allocate from the FrameInfo
  unsigned NumBytes = MFI->getStackSize();
  if (hasFP(MF)) {
    // Get the offset of the stack slot for the EBP register... which is
    // guaranteed to be the last slot by processFunctionBeforeFrameFinalized.
    int EBPOffset = MFI->getObjectOffset(MFI->getObjectIndexBegin())+4;

    if (NumBytes) {   // adjust stack pointer: ESP -= numbytes
      MI= BuildMI(X86::SUB32ri, 1, X86::ESP, MachineOperand::UseAndDef)
            .addZImm(NumBytes);
      MBB.insert(MBBI, MI);
    }

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

  } else {
    if (MFI->hasCalls()) {
      // When we have no frame pointer, we reserve argument space for call sites
      // in the function immediately on entry to the current function.  This
      // eliminates the need for add/sub ESP brackets around call sites.
      //
      NumBytes += MFI->getMaxCallFrameSize();

      // Round the size to a multiple of the alignment (don't forget the 4 byte
      // offset though).
      unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
      NumBytes = ((NumBytes+4)+Align-1)/Align*Align - 4;
    }

    // Update frame info to pretend that this is part of the stack...
    MFI->setStackSize(NumBytes);

    if (NumBytes) {
      // adjust stack pointer: ESP -= numbytes
      MI= BuildMI(X86::SUB32ri, 1, X86::ESP, MachineOperand::UseAndDef)
            .addZImm(NumBytes);
      MBB.insert(MBBI, MI);
    }
  }
}

void X86RegisterInfo::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  MachineInstr *MI;
  assert(MBBI->getOpcode() == X86::RET &&
         "Can only insert epilog into returning blocks");

  if (hasFP(MF)) {
    // Get the offset of the stack slot for the EBP register... which is
    // guaranteed to be the last slot by processFunctionBeforeFrameFinalized.
    int EBPOffset = MFI->getObjectOffset(MFI->getObjectIndexEnd()-1)+4;

    // mov ESP, EBP
    MI = BuildMI(X86::MOV32rr, 1,X86::ESP).addReg(X86::EBP);
    MBB.insert(MBBI, MI);

    // pop EBP
    MI = BuildMI(X86::POP32r, 0, X86::EBP);
    MBB.insert(MBBI, MI);
  } else {
    // Get the number of bytes allocated from the FrameInfo...
    unsigned NumBytes = MFI->getStackSize();

    if (NumBytes) {    // adjust stack pointer back: ESP += numbytes
      MI =BuildMI(X86::ADD32ri, 1, X86::ESP, MachineOperand::UseAndDef)
            .addZImm(NumBytes);
      MBB.insert(MBBI, MI);
    }
  }
}

#include "X86GenRegisterInfo.inc"

const TargetRegisterClass*
X86RegisterInfo::getRegClassForType(const Type* Ty) const {
  switch (Ty->getPrimitiveID()) {
  case Type::LongTyID:
  case Type::ULongTyID: assert(0 && "Long values can't fit in registers!");
  default:              assert(0 && "Invalid type to getClass!");
  case Type::BoolTyID:
  case Type::SByteTyID:
  case Type::UByteTyID:   return &R8Instance;
  case Type::ShortTyID:
  case Type::UShortTyID:  return &R16Instance;
  case Type::IntTyID:
  case Type::UIntTyID:
  case Type::PointerTyID: return &R32Instance;

  case Type::FloatTyID:
  case Type::DoubleTyID: return &RFPInstance;
  }
}
