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
  bool Is64BitMode;
public:
  X86MCCodeEmitter(TargetMachine &tm) 
    : TM(tm), TII(*TM.getInstrInfo()) {
    // FIXME: Get this from the right place.
    Is64BitMode = false;
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

  void EmitDisplacementField(const MCOperand *RelocOp, int DispVal,
                             int64_t Adj, bool IsPCRel, raw_ostream &OS) const;
  
  inline static unsigned char ModRMByte(unsigned Mod, unsigned RegOpcode,
                                        unsigned RM) {
    assert(Mod < 4 && RegOpcode < 8 && RM < 8 && "ModRM Fields out of range!");
    return RM | (RegOpcode << 3) | (Mod << 6);
  }
  
  void EmitRegModRMByte(const MCOperand &ModRMReg, unsigned RegOpcodeFld,
                        raw_ostream &OS) const {
    EmitByte(ModRMByte(3, RegOpcodeFld, GetX86RegNum(ModRMReg)), OS);
  }
  
  void EmitSIBByte(unsigned SS, unsigned Index, unsigned Base,
                   raw_ostream &OS) const {
    // SIB byte is in the same format as the ModRMByte...
    EmitByte(ModRMByte(SS, Index, Base), OS);
  }
  
  
  void EmitMemModRMByte(const MCInst &MI, unsigned Op,
                        unsigned RegOpcodeField, intptr_t PCAdj,
                        raw_ostream &OS) const;
  
  void EncodeInstruction(const MCInst &MI, raw_ostream &OS) const;
  
};

} // end anonymous namespace


MCCodeEmitter *llvm::createX86MCCodeEmitter(const Target &,
                                            TargetMachine &TM) {
  return new X86MCCodeEmitter(TM);
}


/// isDisp8 - Return true if this signed displacement fits in a 8-bit 
/// sign-extended field. 
static bool isDisp8(int Value) {
  return Value == (signed char)Value;
}

void X86MCCodeEmitter::
EmitDisplacementField(const MCOperand *RelocOp, int DispVal,
                      int64_t Adj, bool IsPCRel, raw_ostream &OS) const {
  // If this is a simple integer displacement that doesn't require a relocation,
  // emit it now.
  if (!RelocOp) {
    EmitConstant(DispVal, 4, OS);
    return;
  }
  
  assert(0 && "Reloc not handled yet");
#if 0
  // Otherwise, this is something that requires a relocation.  Emit it as such
  // now.
  unsigned RelocType = Is64BitMode ?
  (IsPCRel ? X86::reloc_pcrel_word : X86::reloc_absolute_word_sext)
  : (IsPIC ? X86::reloc_picrel_word : X86::reloc_absolute_word);
  if (RelocOp->isGlobal()) {
    // In 64-bit static small code model, we could potentially emit absolute.
    // But it's probably not beneficial. If the MCE supports using RIP directly
    // do it, otherwise fallback to absolute (this is determined by IsPCRel). 
    //  89 05 00 00 00 00     mov    %eax,0(%rip)  # PC-relative
    //  89 04 25 00 00 00 00  mov    %eax,0x0      # Absolute
    bool Indirect = gvNeedsNonLazyPtr(*RelocOp, TM);
    emitGlobalAddress(RelocOp->getGlobal(), RelocType, RelocOp->getOffset(),
                      Adj, Indirect);
  } else if (RelocOp->isSymbol()) {
    emitExternalSymbolAddress(RelocOp->getSymbolName(), RelocType);
  } else if (RelocOp->isCPI()) {
    emitConstPoolAddress(RelocOp->getIndex(), RelocType,
                         RelocOp->getOffset(), Adj);
  } else {
    assert(RelocOp->isJTI() && "Unexpected machine operand!");
    emitJumpTableAddress(RelocOp->getIndex(), RelocType, Adj);
  }
#endif
}


void X86MCCodeEmitter::EmitMemModRMByte(const MCInst &MI, unsigned Op,
                                        unsigned RegOpcodeField,
                                        intptr_t PCAdj,
                                        raw_ostream &OS) const {
  const MCOperand &Op3 = MI.getOperand(Op+3);
  int DispVal = 0;
  const MCOperand *DispForReloc = 0;
  
  // Figure out what sort of displacement we have to handle here.
  if (Op3.isImm()) {
    DispVal = Op3.getImm();
  } else {
    assert(0 && "relocatable operand");
#if 0
  if (Op3.isGlobal()) {
    DispForReloc = &Op3;
  } else if (Op3.isSymbol()) {
    DispForReloc = &Op3;
  } else if (Op3.isCPI()) {
    if (!MCE.earlyResolveAddresses() || Is64BitMode || IsPIC) {
      DispForReloc = &Op3;
    } else {
      DispVal += MCE.getConstantPoolEntryAddress(Op3.getIndex());
      DispVal += Op3.getOffset();
    }
  } else {
    assert(Op3.isJTI());
    if (!MCE.earlyResolveAddresses() || Is64BitMode || IsPIC) {
      DispForReloc = &Op3;
    } else {
      DispVal += MCE.getJumpTableEntryAddress(Op3.getIndex());
    }
#endif
  }
  
  const MCOperand &Base     = MI.getOperand(Op);
  const MCOperand &Scale    = MI.getOperand(Op+1);
  const MCOperand &IndexReg = MI.getOperand(Op+2);
  unsigned BaseReg = Base.getReg();

  // FIXME: Eliminate!
  bool IsPCRel = false;

  // Is a SIB byte needed?
  // If no BaseReg, issue a RIP relative instruction only if the MCE can 
  // resolve addresses on-the-fly, otherwise use SIB (Intel Manual 2A, table
  // 2-7) and absolute references.
  if ((!Is64BitMode || DispForReloc || BaseReg != 0) &&
      IndexReg.getReg() == 0 && 
      (BaseReg == X86::RIP || (BaseReg != 0 && BaseReg != X86::ESP))) {
    if (BaseReg == 0 || BaseReg == X86::RIP) {  // Just a displacement?
      // Emit special case [disp32] encoding
      EmitByte(ModRMByte(0, RegOpcodeField, 5), OS);
      EmitDisplacementField(DispForReloc, DispVal, PCAdj, true, OS);
    } else {
      unsigned BaseRegNo = GetX86RegNum(Base);
      if (!DispForReloc && DispVal == 0 && BaseRegNo != N86::EBP) {
        // Emit simple indirect register encoding... [EAX] f.e.
        EmitByte(ModRMByte(0, RegOpcodeField, BaseRegNo), OS);
      } else if (!DispForReloc && isDisp8(DispVal)) {
        // Emit the disp8 encoding... [REG+disp8]
        EmitByte(ModRMByte(1, RegOpcodeField, BaseRegNo), OS);
        EmitConstant(DispVal, 1, OS);
      } else {
        // Emit the most general non-SIB encoding: [REG+disp32]
        EmitByte(ModRMByte(2, RegOpcodeField, BaseRegNo), OS);
        EmitDisplacementField(DispForReloc, DispVal, PCAdj, IsPCRel, OS);
      }
    }
    return;
  }
    
  // We need a SIB byte, so start by outputting the ModR/M byte first
  assert(IndexReg.getReg() != X86::ESP &&
         IndexReg.getReg() != X86::RSP && "Cannot use ESP as index reg!");
  
  bool ForceDisp32 = false;
  bool ForceDisp8  = false;
  if (BaseReg == 0) {
    // If there is no base register, we emit the special case SIB byte with
    // MOD=0, BASE=5, to JUST get the index, scale, and displacement.
    EmitByte(ModRMByte(0, RegOpcodeField, 4), OS);
    ForceDisp32 = true;
  } else if (DispForReloc) {
    // Emit the normal disp32 encoding.
    EmitByte(ModRMByte(2, RegOpcodeField, 4), OS);
    ForceDisp32 = true;
  } else if (DispVal == 0 && BaseReg != X86::EBP) {
    // Emit no displacement ModR/M byte
    EmitByte(ModRMByte(0, RegOpcodeField, 4), OS);
  } else if (isDisp8(DispVal)) {
    // Emit the disp8 encoding.
    EmitByte(ModRMByte(1, RegOpcodeField, 4), OS);
    ForceDisp8 = true;           // Make sure to force 8 bit disp if Base=EBP
  } else {
    // Emit the normal disp32 encoding.
    EmitByte(ModRMByte(2, RegOpcodeField, 4), OS);
  }
  
  // Calculate what the SS field value should be...
  static const unsigned SSTable[] = { ~0, 0, 1, ~0, 2, ~0, ~0, ~0, 3 };
  unsigned SS = SSTable[Scale.getImm()];
  
  if (BaseReg == 0) {
    // Handle the SIB byte for the case where there is no base, see Intel 
    // Manual 2A, table 2-7. The displacement has already been output.
    unsigned IndexRegNo;
    if (IndexReg.getReg())
      IndexRegNo = GetX86RegNum(IndexReg);
    else // Examples: [ESP+1*<noreg>+4] or [scaled idx]+disp32 (MOD=0,BASE=5)
      IndexRegNo = 4;
    EmitSIBByte(SS, IndexRegNo, 5, OS);
  } else {
    unsigned IndexRegNo;
    if (IndexReg.getReg())
      IndexRegNo = GetX86RegNum(IndexReg);
    else
      IndexRegNo = 4;   // For example [ESP+1*<noreg>+4]
    EmitSIBByte(SS, IndexRegNo, GetX86RegNum(Base), OS);
  }
  
  // Do we need to output a displacement?
  if (ForceDisp8)
    EmitConstant(DispVal, 1, OS);
  else if (DispVal != 0 || ForceDisp32)
    EmitDisplacementField(DispForReloc, DispVal, PCAdj, IsPCRel, OS);
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
  
  // FIXME: Can we kill off MRMInitReg??
  
  unsigned char BaseOpcode = X86II::getBaseOpcodeFor(TSFlags);
  switch (TSFlags & X86II::FormMask) {
  default: errs() << "FORM: " << (TSFlags & X86II::FormMask) << "\n";
      assert(0 && "Unknown FormMask value in X86MCCodeEmitter!");
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
      unsigned Size = X86II::getSizeOfImm(TSFlags);
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
                   X86II::getSizeOfImm(TSFlags), OS);
    break;
  
  case X86II::MRMDestMem:
    EmitByte(BaseOpcode, OS);
    EmitMemModRMByte(MI, CurOp,
                     GetX86RegNum(MI.getOperand(CurOp + X86AddrNumOperands)),
                     0, OS);
    CurOp +=  X86AddrNumOperands + 1;
    if (CurOp != NumOps)
      EmitConstant(MI.getOperand(CurOp++).getImm(),
                   X86II::getSizeOfImm(TSFlags), OS);
    break;
      
  case X86II::MRMSrcReg:
    EmitByte(BaseOpcode, OS);
    EmitRegModRMByte(MI.getOperand(CurOp+1), GetX86RegNum(MI.getOperand(CurOp)),
                     OS);
    CurOp += 2;
    if (CurOp != NumOps)
      EmitConstant(MI.getOperand(CurOp++).getImm(),
                   X86II::getSizeOfImm(TSFlags), OS);
    break;
    
  case X86II::MRMSrcMem: {
    EmitByte(BaseOpcode, OS);

    // FIXME: Maybe lea should have its own form?  This is a horrible hack.
    int AddrOperands;
    if (Opcode == X86::LEA64r || Opcode == X86::LEA64_32r ||
        Opcode == X86::LEA16r || Opcode == X86::LEA32r)
      AddrOperands = X86AddrNumOperands - 1; // No segment register
    else
      AddrOperands = X86AddrNumOperands;
    
    // FIXME: What is this actually doing?
    intptr_t PCAdj = (CurOp + AddrOperands + 1 != NumOps) ?
       X86II::getSizeOfImm(TSFlags) : 0;
    
    EmitMemModRMByte(MI, CurOp+1, GetX86RegNum(MI.getOperand(CurOp)),
                     PCAdj, OS);
    CurOp += AddrOperands + 1;
    if (CurOp != NumOps)
      EmitConstant(MI.getOperand(CurOp++).getImm(),
                   X86II::getSizeOfImm(TSFlags), OS);
    break;
  }
      
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
