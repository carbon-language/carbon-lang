//===-- X86/X86MCCodeEmitter.cpp - Convert X86 code to machine code -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the X86MCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "x86-emitter"
#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
class X86MCCodeEmitter : public MCCodeEmitter {
  X86MCCodeEmitter(const X86MCCodeEmitter &); // DO NOT IMPLEMENT
  void operator=(const X86MCCodeEmitter &); // DO NOT IMPLEMENT
  const TargetMachine &TM;
  const TargetInstrInfo &TII;
public:
  X86MCCodeEmitter(TargetMachine &tm) 
    : TM(tm), TII(*TM.getInstrInfo()) {
  }

  ~X86MCCodeEmitter() {}
  
  static unsigned GetX86RegNum(const MCOperand &MO) {
    return X86RegisterInfo::getX86RegNum(MO.getReg());
  }
  
  void EmitByte(unsigned char C, raw_ostream &OS) const {
    OS << (char)C;
  }
  
  void EmitConstant(uint64_t Val, unsigned Size, raw_ostream &OS) const {
    // Output the constant in little endian byte order.
    for (unsigned i = 0; i != Size; ++i) {
      EmitByte(Val & 255, OS);
      Val >>= 8;
    }
  }
  
  inline static unsigned char ModRMByte(unsigned Mod, unsigned RegOpcode,
                                        unsigned RM) {
    assert(Mod < 4 && RegOpcode < 8 && RM < 8 && "ModRM Fields out of range!");
    return RM | (RegOpcode << 3) | (Mod << 6);
  }
  
  void EmitRegModRMByte(const MCOperand &ModRMReg, unsigned RegOpcodeFld,
                        raw_ostream &OS) const {
    EmitByte(ModRMByte(3, RegOpcodeFld, GetX86RegNum(ModRMReg)), OS);
  }
  
  
  void EncodeInstruction(const MCInst &MI, raw_ostream &OS) const;
  
};

} // end anonymous namespace


MCCodeEmitter *llvm::createX86MCCodeEmitter(const Target &,
                                            TargetMachine &TM) {
  return new X86MCCodeEmitter(TM);
}



void X86MCCodeEmitter::
EncodeInstruction(const MCInst &MI, raw_ostream &OS) const {
  unsigned Opcode = MI.getOpcode();
  const TargetInstrDesc &Desc = TII.get(Opcode);
  unsigned TSFlags = Desc.TSFlags;

  // FIXME: We should emit the prefixes in exactly the same order as GAS does,
  // in order to provide diffability.

  // Emit the lock opcode prefix as needed.
  if (TSFlags & X86II::LOCK)
    EmitByte(0xF0, OS);
  
  // Emit segment override opcode prefix as needed.
  switch (TSFlags & X86II::SegOvrMask) {
  default: assert(0 && "Invalid segment!");
  case 0: break;  // No segment override!
  case X86II::FS:
    EmitByte(0x64, OS);
    break;
  case X86II::GS:
    EmitByte(0x65, OS);
    break;
  }
  
  // Emit the repeat opcode prefix as needed.
  if ((TSFlags & X86II::Op0Mask) == X86II::REP)
    EmitByte(0xF3, OS);
  
  // Emit the operand size opcode prefix as needed.
  if (TSFlags & X86II::OpSize)
    EmitByte(0x66, OS);
  
  // Emit the address size opcode prefix as needed.
  if (TSFlags & X86II::AdSize)
    EmitByte(0x67, OS);
  
  bool Need0FPrefix = false;
  switch (TSFlags & X86II::Op0Mask) {
  default: assert(0 && "Invalid prefix!");
  case 0: break;  // No prefix!
  case X86II::REP: break; // already handled.
  case X86II::TB:  // Two-byte opcode prefix
  case X86II::T8:  // 0F 38
  case X86II::TA:  // 0F 3A
    Need0FPrefix = true;
    break;
  case X86II::TF: // F2 0F 38
    EmitByte(0xF2, OS);
    Need0FPrefix = true;
    break;
  case X86II::XS:   // F3 0F
    EmitByte(0xF3, OS);
    Need0FPrefix = true;
    break;
  case X86II::XD:   // F2 0F
    EmitByte(0xF2, OS);
    Need0FPrefix = true;
    break;
  case X86II::D8: EmitByte(0xD8, OS); break;
  case X86II::D9: EmitByte(0xD9, OS); break;
  case X86II::DA: EmitByte(0xDA, OS); break;
  case X86II::DB: EmitByte(0xDB, OS); break;
  case X86II::DC: EmitByte(0xDC, OS); break;
  case X86II::DD: EmitByte(0xDD, OS); break;
  case X86II::DE: EmitByte(0xDE, OS); break;
  case X86II::DF: EmitByte(0xDF, OS); break;
  }
  
  // Handle REX prefix.
#if 0 // FIXME: Add in, also, can this come before F2 etc to simplify emission?
  if (Is64BitMode) {
    if (unsigned REX = X86InstrInfo::determineREX(MI))
      EmitByte(0x40 | REX, OS);
  }
#endif
  
  // 0x0F escape code must be emitted just before the opcode.
  if (Need0FPrefix)
    EmitByte(0x0F, OS);
  
  // FIXME: Pull this up into previous switch if REX can be moved earlier.
  switch (TSFlags & X86II::Op0Mask) {
  case X86II::TF:    // F2 0F 38
  case X86II::T8:    // 0F 38
    EmitByte(0x38, OS);
    break;
  case X86II::TA:    // 0F 3A
    EmitByte(0x3A, OS);
    break;
  }
  
  // If this is a two-address instruction, skip one of the register operands.
  unsigned NumOps = Desc.getNumOperands();
  unsigned CurOp = 0;
  if (NumOps > 1 && Desc.getOperandConstraint(1, TOI::TIED_TO) != -1)
    ++CurOp;
  else if (NumOps > 2 && Desc.getOperandConstraint(NumOps-1, TOI::TIED_TO)== 0)
    // Skip the last source operand that is tied_to the dest reg. e.g. LXADD32
    --NumOps;
  
  unsigned char BaseOpcode = X86InstrInfo::getBaseOpcodeFor(Desc);
  switch (TSFlags & X86II::FormMask) {
  default: assert(0 && "Unknown FormMask value in X86MCCodeEmitter!");
  case X86II::RawFrm: {
    EmitByte(BaseOpcode, OS);
    
    if (CurOp == NumOps)
      break;
    
    assert(0 && "Unimpl RawFrm expr");
    break;
  }
      
  case X86II::AddRegFrm: {
    EmitByte(BaseOpcode + GetX86RegNum(MI.getOperand(CurOp++)),OS);
    if (CurOp == NumOps)
      break;

    const MCOperand &MO1 = MI.getOperand(CurOp++);
    if (MO1.isImm()) {
      unsigned Size = X86InstrInfo::sizeOfImm(&Desc);
      EmitConstant(MO1.getImm(), Size, OS);
      break;
    }

    assert(0 && "Unimpl AddRegFrm expr");
    break;
  }
      
  case X86II::MRMDestReg:
    EmitByte(BaseOpcode, OS);
    EmitRegModRMByte(MI.getOperand(CurOp),
                     GetX86RegNum(MI.getOperand(CurOp+1)), OS);
    CurOp += 2;
    if (CurOp != NumOps)
      EmitConstant(MI.getOperand(CurOp++).getImm(),
                   X86InstrInfo::sizeOfImm(&Desc), OS);
    break;
  }
  
#ifndef NDEBUG
  if (!Desc.isVariadic() && CurOp != NumOps) {
    errs() << "Cannot encode all operands of: ";
    MI.dump();
    errs() << '\n';
    abort();
  }
#endif
}
