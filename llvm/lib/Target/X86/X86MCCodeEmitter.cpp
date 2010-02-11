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

// FIXME: This should move to a header.
namespace llvm {
namespace X86 {
enum Fixups {
  reloc_pcrel_word = FirstTargetFixupKind,
  reloc_picrel_word,
  reloc_absolute_word,
  reloc_absolute_word_sext,
  reloc_absolute_dword
};
}
}

namespace {
class X86MCCodeEmitter : public MCCodeEmitter {
  X86MCCodeEmitter(const X86MCCodeEmitter &); // DO NOT IMPLEMENT
  void operator=(const X86MCCodeEmitter &); // DO NOT IMPLEMENT
  const TargetMachine &TM;
  const TargetInstrInfo &TII;
  bool Is64BitMode;
public:
  X86MCCodeEmitter(TargetMachine &tm, bool is64Bit) 
    : TM(tm), TII(*TM.getInstrInfo()) {
    Is64BitMode = is64Bit;
  }

  ~X86MCCodeEmitter() {}

  unsigned getNumFixupKinds() const {
    return 5;
  }

  MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const {
    static MCFixupKindInfo Infos[] = {
      { "reloc_pcrel_word", 0, 4 * 8 },
      { "reloc_picrel_word", 0, 4 * 8 },
      { "reloc_absolute_word", 0, 4 * 8 },
      { "reloc_absolute_word_sext", 0, 4 * 8 },
      { "reloc_absolute_dword", 0, 8 * 8 }
    };

    assert(Kind >= FirstTargetFixupKind && Kind < MaxTargetFixupKind &&
           "Invalid kind!");
    return Infos[Kind - FirstTargetFixupKind];
  }
  
  static unsigned GetX86RegNum(const MCOperand &MO) {
    return X86RegisterInfo::getX86RegNum(MO.getReg());
  }
  
  void EmitByte(unsigned char C, unsigned &CurByte, raw_ostream &OS) const {
    OS << (char)C;
    ++CurByte;
  }
  
  void EmitConstant(uint64_t Val, unsigned Size, unsigned &CurByte,
                    raw_ostream &OS) const {
    // Output the constant in little endian byte order.
    for (unsigned i = 0; i != Size; ++i) {
      EmitByte(Val & 255, CurByte, OS);
      Val >>= 8;
    }
  }

  void EmitDisplacementField(const MCOperand &Disp, unsigned &CurByte,
                             raw_ostream &OS,
                             SmallVectorImpl<MCFixup> &Fixups) const;
  
  inline static unsigned char ModRMByte(unsigned Mod, unsigned RegOpcode,
                                        unsigned RM) {
    assert(Mod < 4 && RegOpcode < 8 && RM < 8 && "ModRM Fields out of range!");
    return RM | (RegOpcode << 3) | (Mod << 6);
  }
  
  void EmitRegModRMByte(const MCOperand &ModRMReg, unsigned RegOpcodeFld,
                        unsigned &CurByte, raw_ostream &OS) const {
    EmitByte(ModRMByte(3, RegOpcodeFld, GetX86RegNum(ModRMReg)), CurByte, OS);
  }
  
  void EmitSIBByte(unsigned SS, unsigned Index, unsigned Base,
                   unsigned &CurByte, raw_ostream &OS) const {
    // SIB byte is in the same format as the ModRMByte.
    EmitByte(ModRMByte(SS, Index, Base), CurByte, OS);
  }
  
  
  void EmitMemModRMByte(const MCInst &MI, unsigned Op,
                        unsigned RegOpcodeField, 
                        unsigned &CurByte, raw_ostream &OS,
                        SmallVectorImpl<MCFixup> &Fixups) const;
  
  void EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups) const;
  
};

} // end anonymous namespace


MCCodeEmitter *llvm::createX86_32MCCodeEmitter(const Target &,
                                               TargetMachine &TM) {
  return new X86MCCodeEmitter(TM, false);
}

MCCodeEmitter *llvm::createX86_64MCCodeEmitter(const Target &,
                                               TargetMachine &TM) {
  return new X86MCCodeEmitter(TM, true);
}


/// isDisp8 - Return true if this signed displacement fits in a 8-bit 
/// sign-extended field. 
static bool isDisp8(int Value) {
  return Value == (signed char)Value;
}

void X86MCCodeEmitter::
EmitDisplacementField(const MCOperand &DispOp,
                      unsigned &CurByte, raw_ostream &OS,
                      SmallVectorImpl<MCFixup> &Fixups) const {
  // If this is a simple integer displacement that doesn't require a relocation,
  // emit it now.
  if (DispOp.isImm()) {
    EmitConstant(DispOp.getImm(), 4, CurByte, OS);
    return;
  }

#if 0
  // Otherwise, this is something that requires a relocation.  Emit it as such
  // now.
  unsigned RelocType = Is64BitMode ?
  (IsPCRel ? X86::reloc_pcrel_word : X86::reloc_absolute_word_sext)
  : (IsPIC ? X86::reloc_picrel_word : X86::reloc_absolute_word);
#endif
  
  // Emit a symbolic constant as a fixup and 4 zeros.
  Fixups.push_back(MCFixup::Create(CurByte, DispOp.getExpr(),
                                   MCFixupKind(X86::reloc_absolute_word)));
  EmitConstant(0, 4, CurByte, OS);
}


void X86MCCodeEmitter::EmitMemModRMByte(const MCInst &MI, unsigned Op,
                                        unsigned RegOpcodeField,
                                        unsigned &CurByte,
                                        raw_ostream &OS,
                                        SmallVectorImpl<MCFixup> &Fixups) const{
  const MCOperand &Disp     = MI.getOperand(Op+3);
  const MCOperand &Base     = MI.getOperand(Op);
  const MCOperand &Scale    = MI.getOperand(Op+1);
  const MCOperand &IndexReg = MI.getOperand(Op+2);
  unsigned BaseReg = Base.getReg();

  // Determine whether a SIB byte is needed.
  // If no BaseReg, issue a RIP relative instruction only if the MCE can 
  // resolve addresses on-the-fly, otherwise use SIB (Intel Manual 2A, table
  // 2-7) and absolute references.
  if (// The SIB byte must be used if there is an index register.
      IndexReg.getReg() == 0 && 
      // The SIB byte must be used if the base is ESP/RSP.
      BaseReg != X86::ESP && BaseReg != X86::RSP &&
      // If there is no base register and we're in 64-bit mode, we need a SIB
      // byte to emit an addr that is just 'disp32' (the non-RIP relative form).
      (!Is64BitMode || BaseReg != 0)) {

    if (BaseReg == 0 ||          // [disp32]     in X86-32 mode
        BaseReg == X86::RIP) {   // [disp32+RIP] in X86-64 mode
      EmitByte(ModRMByte(0, RegOpcodeField, 5), CurByte, OS);
      EmitDisplacementField(Disp, CurByte, OS, Fixups);
      return;
    }
    
    unsigned BaseRegNo = GetX86RegNum(Base);

    // If the base is not EBP/ESP and there is no displacement, use simple
    // indirect register encoding, this handles addresses like [EAX].  The
    // encoding for [EBP] with no displacement means [disp32] so we handle it
    // by emitting a displacement of 0 below.
    if (Disp.isImm() && Disp.getImm() == 0 && BaseRegNo != N86::EBP) {
      EmitByte(ModRMByte(0, RegOpcodeField, BaseRegNo), CurByte, OS);
      return;
    }
    
    // Otherwise, if the displacement fits in a byte, encode as [REG+disp8].
    if (Disp.isImm() && isDisp8(Disp.getImm())) {
      EmitByte(ModRMByte(1, RegOpcodeField, BaseRegNo), CurByte, OS);
      EmitConstant(Disp.getImm(), 1, CurByte, OS);
      return;
    }
    
    // Otherwise, emit the most general non-SIB encoding: [REG+disp32]
    EmitByte(ModRMByte(2, RegOpcodeField, BaseRegNo), CurByte, OS);
    EmitDisplacementField(Disp, CurByte, OS, Fixups);
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
    EmitByte(ModRMByte(0, RegOpcodeField, 4), CurByte, OS);
    ForceDisp32 = true;
  } else if (!Disp.isImm()) {
    // Emit the normal disp32 encoding.
    EmitByte(ModRMByte(2, RegOpcodeField, 4), CurByte, OS);
    ForceDisp32 = true;
  } else if (Disp.getImm() == 0 && BaseReg != X86::EBP) {
    // Emit no displacement ModR/M byte
    EmitByte(ModRMByte(0, RegOpcodeField, 4), CurByte, OS);
  } else if (isDisp8(Disp.getImm())) {
    // Emit the disp8 encoding.
    EmitByte(ModRMByte(1, RegOpcodeField, 4), CurByte, OS);
    ForceDisp8 = true;           // Make sure to force 8 bit disp if Base=EBP
  } else {
    // Emit the normal disp32 encoding.
    EmitByte(ModRMByte(2, RegOpcodeField, 4), CurByte, OS);
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
    EmitSIBByte(SS, IndexRegNo, 5, CurByte, OS);
  } else {
    unsigned IndexRegNo;
    if (IndexReg.getReg())
      IndexRegNo = GetX86RegNum(IndexReg);
    else
      IndexRegNo = 4;   // For example [ESP+1*<noreg>+4]
    EmitSIBByte(SS, IndexRegNo, GetX86RegNum(Base), CurByte, OS);
  }
  
  // Do we need to output a displacement?
  if (ForceDisp8)
    EmitConstant(Disp.getImm(), 1, CurByte, OS);
  else if (ForceDisp32 || Disp.getImm() != 0)
    EmitDisplacementField(Disp, CurByte, OS, Fixups);
}

/// DetermineREXPrefix - Determine if the MCInst has to be encoded with a X86-64
/// REX prefix which specifies 1) 64-bit instructions, 2) non-default operand
/// size, and 3) use of X86-64 extended registers.
static unsigned DetermineREXPrefix(const MCInst &MI, unsigned TSFlags,
                                   const TargetInstrDesc &Desc) {
  unsigned REX = 0;
  
  // Pseudo instructions do not need REX prefix byte.
  if ((TSFlags & X86II::FormMask) == X86II::Pseudo)
    return 0;
  if (TSFlags & X86II::REX_W)
    REX |= 1 << 3;
  
  if (MI.getNumOperands() == 0) return REX;
  
  unsigned NumOps = MI.getNumOperands();
  // FIXME: MCInst should explicitize the two-addrness.
  bool isTwoAddr = NumOps > 1 &&
                      Desc.getOperandConstraint(1, TOI::TIED_TO) != -1;
  
  // If it accesses SPL, BPL, SIL, or DIL, then it requires a 0x40 REX prefix.
  unsigned i = isTwoAddr ? 1 : 0;
  for (; i != NumOps; ++i) {
    const MCOperand &MO = MI.getOperand(i);
    if (!MO.isReg()) continue;
    unsigned Reg = MO.getReg();
    if (!X86InstrInfo::isX86_64NonExtLowByteReg(Reg)) continue;
    // FIXME: The caller of DetermineREXPrefix slaps this prefix onto anything
    // that returns non-zero.
    REX |= 0x40;
    break;
  }
  
  switch (TSFlags & X86II::FormMask) {
  case X86II::MRMInitReg: assert(0 && "FIXME: Remove this!");
  case X86II::MRMSrcReg:
    if (MI.getOperand(0).isReg() &&
        X86InstrInfo::isX86_64ExtendedReg(MI.getOperand(0).getReg()))
      REX |= 1 << 2;
    i = isTwoAddr ? 2 : 1;
    for (; i != NumOps; ++i) {
      const MCOperand &MO = MI.getOperand(i);
      if (MO.isReg() && X86InstrInfo::isX86_64ExtendedReg(MO.getReg()))
        REX |= 1 << 0;
    }
    break;
  case X86II::MRMSrcMem: {
    if (MI.getOperand(0).isReg() &&
        X86InstrInfo::isX86_64ExtendedReg(MI.getOperand(0).getReg()))
      REX |= 1 << 2;
    unsigned Bit = 0;
    i = isTwoAddr ? 2 : 1;
    for (; i != NumOps; ++i) {
      const MCOperand &MO = MI.getOperand(i);
      if (MO.isReg()) {
        if (X86InstrInfo::isX86_64ExtendedReg(MO.getReg()))
          REX |= 1 << Bit;
        Bit++;
      }
    }
    break;
  }
  case X86II::MRM0m: case X86II::MRM1m:
  case X86II::MRM2m: case X86II::MRM3m:
  case X86II::MRM4m: case X86II::MRM5m:
  case X86II::MRM6m: case X86II::MRM7m:
  case X86II::MRMDestMem: {
    unsigned e = (isTwoAddr ? X86AddrNumOperands+1 : X86AddrNumOperands);
    i = isTwoAddr ? 1 : 0;
    if (NumOps > e && MI.getOperand(e).isReg() &&
        X86InstrInfo::isX86_64ExtendedReg(MI.getOperand(e).getReg()))
      REX |= 1 << 2;
    unsigned Bit = 0;
    for (; i != e; ++i) {
      const MCOperand &MO = MI.getOperand(i);
      if (MO.isReg()) {
        if (X86InstrInfo::isX86_64ExtendedReg(MO.getReg()))
          REX |= 1 << Bit;
        Bit++;
      }
    }
    break;
  }
  default:
    if (MI.getOperand(0).isReg() &&
        X86InstrInfo::isX86_64ExtendedReg(MI.getOperand(0).getReg()))
      REX |= 1 << 0;
    i = isTwoAddr ? 2 : 1;
    for (unsigned e = NumOps; i != e; ++i) {
      const MCOperand &MO = MI.getOperand(i);
      if (MO.isReg() && X86InstrInfo::isX86_64ExtendedReg(MO.getReg()))
        REX |= 1 << 2;
    }
    break;
  }
  return REX;
}

void X86MCCodeEmitter::
EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                  SmallVectorImpl<MCFixup> &Fixups) const {
  unsigned Opcode = MI.getOpcode();
  const TargetInstrDesc &Desc = TII.get(Opcode);
  unsigned TSFlags = Desc.TSFlags;

  // Keep track of the current byte being emitted.
  unsigned CurByte = 0;
  
  // FIXME: We should emit the prefixes in exactly the same order as GAS does,
  // in order to provide diffability.

  // Emit the lock opcode prefix as needed.
  if (TSFlags & X86II::LOCK)
    EmitByte(0xF0, CurByte, OS);
  
  // Emit segment override opcode prefix as needed.
  switch (TSFlags & X86II::SegOvrMask) {
  default: assert(0 && "Invalid segment!");
  case 0: break;  // No segment override!
  case X86II::FS:
    EmitByte(0x64, CurByte, OS);
    break;
  case X86II::GS:
    EmitByte(0x65, CurByte, OS);
    break;
  }
  
  // Emit the repeat opcode prefix as needed.
  if ((TSFlags & X86II::Op0Mask) == X86II::REP)
    EmitByte(0xF3, CurByte, OS);
  
  // Emit the operand size opcode prefix as needed.
  if (TSFlags & X86II::OpSize)
    EmitByte(0x66, CurByte, OS);
  
  // Emit the address size opcode prefix as needed.
  if (TSFlags & X86II::AdSize)
    EmitByte(0x67, CurByte, OS);
  
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
    EmitByte(0xF2, CurByte, OS);
    Need0FPrefix = true;
    break;
  case X86II::XS:   // F3 0F
    EmitByte(0xF3, CurByte, OS);
    Need0FPrefix = true;
    break;
  case X86II::XD:   // F2 0F
    EmitByte(0xF2, CurByte, OS);
    Need0FPrefix = true;
    break;
  case X86II::D8: EmitByte(0xD8, CurByte, OS); break;
  case X86II::D9: EmitByte(0xD9, CurByte, OS); break;
  case X86II::DA: EmitByte(0xDA, CurByte, OS); break;
  case X86II::DB: EmitByte(0xDB, CurByte, OS); break;
  case X86II::DC: EmitByte(0xDC, CurByte, OS); break;
  case X86II::DD: EmitByte(0xDD, CurByte, OS); break;
  case X86II::DE: EmitByte(0xDE, CurByte, OS); break;
  case X86II::DF: EmitByte(0xDF, CurByte, OS); break;
  }
  
  // Handle REX prefix.
  // FIXME: Can this come before F2 etc to simplify emission?
  if (Is64BitMode) {
    if (unsigned REX = DetermineREXPrefix(MI, TSFlags, Desc))
      EmitByte(0x40 | REX, CurByte, OS);
  }
  
  // 0x0F escape code must be emitted just before the opcode.
  if (Need0FPrefix)
    EmitByte(0x0F, CurByte, OS);
  
  // FIXME: Pull this up into previous switch if REX can be moved earlier.
  switch (TSFlags & X86II::Op0Mask) {
  case X86II::TF:    // F2 0F 38
  case X86II::T8:    // 0F 38
    EmitByte(0x38, CurByte, OS);
    break;
  case X86II::TA:    // 0F 3A
    EmitByte(0x3A, CurByte, OS);
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
  
  unsigned char BaseOpcode = X86II::getBaseOpcodeFor(TSFlags);
  switch (TSFlags & X86II::FormMask) {
  case X86II::MRMInitReg:
    assert(0 && "FIXME: Remove this form when the JIT moves to MCCodeEmitter!");
  default: errs() << "FORM: " << (TSFlags & X86II::FormMask) << "\n";
      assert(0 && "Unknown FormMask value in X86MCCodeEmitter!");
  case X86II::RawFrm: {
    EmitByte(BaseOpcode, CurByte, OS);
    
    if (CurOp == NumOps)
      break;
    
    assert(0 && "Unimpl RawFrm expr");
    break;
  }
      
  case X86II::AddRegFrm: {
    EmitByte(BaseOpcode + GetX86RegNum(MI.getOperand(CurOp++)), CurByte, OS);
    if (CurOp == NumOps)
      break;

    const MCOperand &MO1 = MI.getOperand(CurOp++);
    if (MO1.isImm()) {
      unsigned Size = X86II::getSizeOfImm(TSFlags);
      EmitConstant(MO1.getImm(), Size, CurByte, OS);
      break;
    }

    assert(0 && "Unimpl AddRegFrm expr");
    break;
  }
      
  case X86II::MRMDestReg:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitRegModRMByte(MI.getOperand(CurOp),
                     GetX86RegNum(MI.getOperand(CurOp+1)), CurByte, OS);
    CurOp += 2;
    if (CurOp != NumOps)
      EmitConstant(MI.getOperand(CurOp++).getImm(),
                   X86II::getSizeOfImm(TSFlags), CurByte, OS);
    break;
  
  case X86II::MRMDestMem:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitMemModRMByte(MI, CurOp,
                     GetX86RegNum(MI.getOperand(CurOp + X86AddrNumOperands)),
                     CurByte, OS, Fixups);
    CurOp += X86AddrNumOperands + 1;
    if (CurOp != NumOps)
      EmitConstant(MI.getOperand(CurOp++).getImm(),
                   X86II::getSizeOfImm(TSFlags), CurByte, OS);
    break;
      
  case X86II::MRMSrcReg:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitRegModRMByte(MI.getOperand(CurOp+1), GetX86RegNum(MI.getOperand(CurOp)),
                     CurByte, OS);
    CurOp += 2;
    if (CurOp != NumOps)
      EmitConstant(MI.getOperand(CurOp++).getImm(),
                   X86II::getSizeOfImm(TSFlags), CurByte, OS);
    break;
    
  case X86II::MRMSrcMem: {
    EmitByte(BaseOpcode, CurByte, OS);

    // FIXME: Maybe lea should have its own form?  This is a horrible hack.
    int AddrOperands;
    if (Opcode == X86::LEA64r || Opcode == X86::LEA64_32r ||
        Opcode == X86::LEA16r || Opcode == X86::LEA32r)
      AddrOperands = X86AddrNumOperands - 1; // No segment register
    else
      AddrOperands = X86AddrNumOperands;
    
    EmitMemModRMByte(MI, CurOp+1, GetX86RegNum(MI.getOperand(CurOp)),
                     CurByte, OS, Fixups);
    CurOp += AddrOperands + 1;
    if (CurOp != NumOps)
      EmitConstant(MI.getOperand(CurOp++).getImm(),
                   X86II::getSizeOfImm(TSFlags), CurByte, OS);
    break;
  }

  case X86II::MRM0r: case X86II::MRM1r:
  case X86II::MRM2r: case X86II::MRM3r:
  case X86II::MRM4r: case X86II::MRM5r:
  case X86II::MRM6r: case X86II::MRM7r: {
    EmitByte(BaseOpcode, CurByte, OS);

    // Special handling of lfence, mfence, monitor, and mwait.
    // FIXME: This is terrible, they should get proper encoding bits in TSFlags.
    if (Opcode == X86::LFENCE || Opcode == X86::MFENCE ||
        Opcode == X86::MONITOR || Opcode == X86::MWAIT) {
      EmitByte(ModRMByte(3, (TSFlags & X86II::FormMask)-X86II::MRM0r, 0),
               CurByte, OS);

      switch (Opcode) {
      default: break;
      case X86::MONITOR: EmitByte(0xC8, CurByte, OS); break;
      case X86::MWAIT:   EmitByte(0xC9, CurByte, OS); break;
      }
    } else {
      EmitRegModRMByte(MI.getOperand(CurOp++),
                       (TSFlags & X86II::FormMask)-X86II::MRM0r,
                       CurByte, OS);
    }

    if (CurOp == NumOps)
      break;
    
    const MCOperand &MO1 = MI.getOperand(CurOp++);
    if (MO1.isImm()) {
      EmitConstant(MO1.getImm(), X86II::getSizeOfImm(TSFlags), CurByte, OS);
      break;
    }

    assert(0 && "relo unimpl");
#if 0
    unsigned rt = Is64BitMode ? X86::reloc_pcrel_word
      : (IsPIC ? X86::reloc_picrel_word : X86::reloc_absolute_word);
    if (Opcode == X86::MOV64ri32)
      rt = X86::reloc_absolute_word_sext;  // FIXME: add X86II flag?
    if (MO1.isGlobal()) {
      bool Indirect = gvNeedsNonLazyPtr(MO1, TM);
      emitGlobalAddress(MO1.getGlobal(), rt, MO1.getOffset(), 0,
                        Indirect);
    } else if (MO1.isSymbol())
      emitExternalSymbolAddress(MO1.getSymbolName(), rt);
    else if (MO1.isCPI())
      emitConstPoolAddress(MO1.getIndex(), rt);
    else if (MO1.isJTI())
      emitJumpTableAddress(MO1.getIndex(), rt);
    break;
#endif
  }
  case X86II::MRM0m: case X86II::MRM1m:
  case X86II::MRM2m: case X86II::MRM3m:
  case X86II::MRM4m: case X86II::MRM5m:
  case X86II::MRM6m: case X86II::MRM7m: {
    EmitByte(BaseOpcode, CurByte, OS);
    EmitMemModRMByte(MI, CurOp, (TSFlags & X86II::FormMask)-X86II::MRM0m,
                     CurByte, OS, Fixups);
    CurOp += X86AddrNumOperands;
    
    if (CurOp == NumOps)
      break;
    
    const MCOperand &MO = MI.getOperand(CurOp++);
    if (MO.isImm()) {
      EmitConstant(MO.getImm(), X86II::getSizeOfImm(TSFlags), CurByte, OS);
      break;
    }
    
    assert(0 && "relo not handled");
#if 0
    unsigned rt = Is64BitMode ? X86::reloc_pcrel_word
    : (IsPIC ? X86::reloc_picrel_word : X86::reloc_absolute_word);
    if (Opcode == X86::MOV64mi32)
      rt = X86::reloc_absolute_word_sext;  // FIXME: add X86II flag?
    if (MO.isGlobal()) {
      bool Indirect = gvNeedsNonLazyPtr(MO, TM);
      emitGlobalAddress(MO.getGlobal(), rt, MO.getOffset(), 0,
                        Indirect);
    } else if (MO.isSymbol())
      emitExternalSymbolAddress(MO.getSymbolName(), rt);
    else if (MO.isCPI())
      emitConstPoolAddress(MO.getIndex(), rt);
    else if (MO.isJTI())
      emitJumpTableAddress(MO.getIndex(), rt);
#endif
    break;
  }
  }
  
#ifndef NDEBUG
  // FIXME: Verify.
  if (/*!Desc.isVariadic() &&*/ CurOp != NumOps) {
    errs() << "Cannot encode all operands of: ";
    MI.dump();
    errs() << '\n';
    abort();
  }
#endif
}
