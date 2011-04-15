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

#define DEBUG_TYPE "mccodeemitter"
#include "X86.h"
#include "X86InstrInfo.h"
#include "X86FixupKinds.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
class X86MCCodeEmitter : public MCCodeEmitter {
  X86MCCodeEmitter(const X86MCCodeEmitter &); // DO NOT IMPLEMENT
  void operator=(const X86MCCodeEmitter &); // DO NOT IMPLEMENT
  const TargetMachine &TM;
  const TargetInstrInfo &TII;
  MCContext &Ctx;
  bool Is64BitMode;
public:
  X86MCCodeEmitter(TargetMachine &tm, MCContext &ctx, bool is64Bit)
    : TM(tm), TII(*TM.getInstrInfo()), Ctx(ctx) {
    Is64BitMode = is64Bit;
  }

  ~X86MCCodeEmitter() {}

  static unsigned GetX86RegNum(const MCOperand &MO) {
    return X86RegisterInfo::getX86RegNum(MO.getReg());
  }

  // On regular x86, both XMM0-XMM7 and XMM8-XMM15 are encoded in the range
  // 0-7 and the difference between the 2 groups is given by the REX prefix.
  // In the VEX prefix, registers are seen sequencially from 0-15 and encoded
  // in 1's complement form, example:
  //
  //  ModRM field => XMM9 => 1
  //  VEX.VVVV    => XMM9 => ~9
  //
  // See table 4-35 of Intel AVX Programming Reference for details.
  static unsigned char getVEXRegisterEncoding(const MCInst &MI,
                                              unsigned OpNum) {
    unsigned SrcReg = MI.getOperand(OpNum).getReg();
    unsigned SrcRegNum = GetX86RegNum(MI.getOperand(OpNum));
    if ((SrcReg >= X86::XMM8 && SrcReg <= X86::XMM15) ||
        (SrcReg >= X86::YMM8 && SrcReg <= X86::YMM15))
      SrcRegNum += 8;

    // The registers represented through VEX_VVVV should
    // be encoded in 1's complement form.
    return (~SrcRegNum) & 0xf;
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

  void EmitImmediate(const MCOperand &Disp,
                     unsigned ImmSize, MCFixupKind FixupKind,
                     unsigned &CurByte, raw_ostream &OS,
                     SmallVectorImpl<MCFixup> &Fixups,
                     int ImmOffset = 0) const;

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
                        uint64_t TSFlags, unsigned &CurByte, raw_ostream &OS,
                        SmallVectorImpl<MCFixup> &Fixups) const;

  void EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups) const;

  void EmitVEXOpcodePrefix(uint64_t TSFlags, unsigned &CurByte, int MemOperand,
                           const MCInst &MI, const TargetInstrDesc &Desc,
                           raw_ostream &OS) const;

  void EmitSegmentOverridePrefix(uint64_t TSFlags, unsigned &CurByte,
                                 int MemOperand, const MCInst &MI,
                                 raw_ostream &OS) const;

  void EmitOpcodePrefix(uint64_t TSFlags, unsigned &CurByte, int MemOperand,
                        const MCInst &MI, const TargetInstrDesc &Desc,
                        raw_ostream &OS) const;
};

} // end anonymous namespace


MCCodeEmitter *llvm::createX86_32MCCodeEmitter(const Target &,
                                               TargetMachine &TM,
                                               MCContext &Ctx) {
  return new X86MCCodeEmitter(TM, Ctx, false);
}

MCCodeEmitter *llvm::createX86_64MCCodeEmitter(const Target &,
                                               TargetMachine &TM,
                                               MCContext &Ctx) {
  return new X86MCCodeEmitter(TM, Ctx, true);
}

/// isDisp8 - Return true if this signed displacement fits in a 8-bit
/// sign-extended field.
static bool isDisp8(int Value) {
  return Value == (signed char)Value;
}

/// getImmFixupKind - Return the appropriate fixup kind to use for an immediate
/// in an instruction with the specified TSFlags.
static MCFixupKind getImmFixupKind(uint64_t TSFlags) {
  unsigned Size = X86II::getSizeOfImm(TSFlags);
  bool isPCRel = X86II::isImmPCRel(TSFlags);

  return MCFixup::getKindForSize(Size, isPCRel);
}

/// Is32BitMemOperand - Return true if the specified instruction with a memory
/// operand should emit the 0x67 prefix byte in 64-bit mode due to a 32-bit
/// memory operand.  Op specifies the operand # of the memoperand.
static bool Is32BitMemOperand(const MCInst &MI, unsigned Op) {
  const MCOperand &BaseReg  = MI.getOperand(Op+X86::AddrBaseReg);
  const MCOperand &IndexReg = MI.getOperand(Op+X86::AddrIndexReg);
  
  if ((BaseReg.getReg() != 0 && X86::GR32RegClass.contains(BaseReg.getReg())) ||
      (IndexReg.getReg() != 0 && X86::GR32RegClass.contains(IndexReg.getReg())))
    return true;
  return false;
}

/// StartsWithGlobalOffsetTable - Return true for the simple cases where this
/// expression starts with _GLOBAL_OFFSET_TABLE_. This is a needed to support
/// PIC on ELF i386 as that symbol is magic. We check only simple case that
/// are know to be used: _GLOBAL_OFFSET_TABLE_ by itself or at the start
/// of a binary expression.
static bool StartsWithGlobalOffsetTable(const MCExpr *Expr) {
  if (Expr->getKind() == MCExpr::Binary) {
    const MCBinaryExpr *BE = static_cast<const MCBinaryExpr *>(Expr);
    Expr = BE->getLHS();
  }

  if (Expr->getKind() != MCExpr::SymbolRef)
    return false;

  const MCSymbolRefExpr *Ref = static_cast<const MCSymbolRefExpr*>(Expr);
  const MCSymbol &S = Ref->getSymbol();
  return S.getName() == "_GLOBAL_OFFSET_TABLE_";
}

void X86MCCodeEmitter::
EmitImmediate(const MCOperand &DispOp, unsigned Size, MCFixupKind FixupKind,
              unsigned &CurByte, raw_ostream &OS,
              SmallVectorImpl<MCFixup> &Fixups, int ImmOffset) const {
  const MCExpr *Expr = NULL;
  if (DispOp.isImm()) {
    // If this is a simple integer displacement that doesn't require a relocation,
    // emit it now.
    if (FixupKind != FK_PCRel_1 &&
	FixupKind != FK_PCRel_2 &&
	FixupKind != FK_PCRel_4) {
      EmitConstant(DispOp.getImm()+ImmOffset, Size, CurByte, OS);
      return;
    }
    Expr = MCConstantExpr::Create(DispOp.getImm(), Ctx);
  } else {
    Expr = DispOp.getExpr();
  }

  // If we have an immoffset, add it to the expression.
  if (FixupKind == FK_Data_4 && StartsWithGlobalOffsetTable(Expr)) {
    assert(ImmOffset == 0);

    FixupKind = MCFixupKind(X86::reloc_global_offset_table);
    ImmOffset = CurByte;
  }

  // If the fixup is pc-relative, we need to bias the value to be relative to
  // the start of the field, not the end of the field.
  if (FixupKind == FK_PCRel_4 ||
      FixupKind == MCFixupKind(X86::reloc_riprel_4byte) ||
      FixupKind == MCFixupKind(X86::reloc_riprel_4byte_movq_load))
    ImmOffset -= 4;
  if (FixupKind == FK_PCRel_2)
    ImmOffset -= 2;
  if (FixupKind == FK_PCRel_1)
    ImmOffset -= 1;

  if (ImmOffset)
    Expr = MCBinaryExpr::CreateAdd(Expr, MCConstantExpr::Create(ImmOffset, Ctx),
                                   Ctx);

  // Emit a symbolic constant as a fixup and 4 zeros.
  Fixups.push_back(MCFixup::Create(CurByte, Expr, FixupKind));
  EmitConstant(0, Size, CurByte, OS);
}

void X86MCCodeEmitter::EmitMemModRMByte(const MCInst &MI, unsigned Op,
                                        unsigned RegOpcodeField,
                                        uint64_t TSFlags, unsigned &CurByte,
                                        raw_ostream &OS,
                                        SmallVectorImpl<MCFixup> &Fixups) const{
  const MCOperand &Disp     = MI.getOperand(Op+X86::AddrDisp);
  const MCOperand &Base     = MI.getOperand(Op+X86::AddrBaseReg);
  const MCOperand &Scale    = MI.getOperand(Op+X86::AddrScaleAmt);
  const MCOperand &IndexReg = MI.getOperand(Op+X86::AddrIndexReg);
  unsigned BaseReg = Base.getReg();

  // Handle %rip relative addressing.
  if (BaseReg == X86::RIP) {    // [disp32+RIP] in X86-64 mode
    assert(Is64BitMode && "Rip-relative addressing requires 64-bit mode");
    assert(IndexReg.getReg() == 0 && "Invalid rip-relative address");
    EmitByte(ModRMByte(0, RegOpcodeField, 5), CurByte, OS);

    unsigned FixupKind = X86::reloc_riprel_4byte;

    // movq loads are handled with a special relocation form which allows the
    // linker to eliminate some loads for GOT references which end up in the
    // same linkage unit.
    if (MI.getOpcode() == X86::MOV64rm)
      FixupKind = X86::reloc_riprel_4byte_movq_load;

    // rip-relative addressing is actually relative to the *next* instruction.
    // Since an immediate can follow the mod/rm byte for an instruction, this
    // means that we need to bias the immediate field of the instruction with
    // the size of the immediate field.  If we have this case, add it into the
    // expression to emit.
    int ImmSize = X86II::hasImm(TSFlags) ? X86II::getSizeOfImm(TSFlags) : 0;

    EmitImmediate(Disp, 4, MCFixupKind(FixupKind),
                  CurByte, OS, Fixups, -ImmSize);
    return;
  }

  unsigned BaseRegNo = BaseReg ? GetX86RegNum(Base) : -1U;

  // Determine whether a SIB byte is needed.
  // If no BaseReg, issue a RIP relative instruction only if the MCE can
  // resolve addresses on-the-fly, otherwise use SIB (Intel Manual 2A, table
  // 2-7) and absolute references.

  if (// The SIB byte must be used if there is an index register.
      IndexReg.getReg() == 0 &&
      // The SIB byte must be used if the base is ESP/RSP/R12, all of which
      // encode to an R/M value of 4, which indicates that a SIB byte is
      // present.
      BaseRegNo != N86::ESP &&
      // If there is no base register and we're in 64-bit mode, we need a SIB
      // byte to emit an addr that is just 'disp32' (the non-RIP relative form).
      (!Is64BitMode || BaseReg != 0)) {

    if (BaseReg == 0) {          // [disp32]     in X86-32 mode
      EmitByte(ModRMByte(0, RegOpcodeField, 5), CurByte, OS);
      EmitImmediate(Disp, 4, FK_Data_4, CurByte, OS, Fixups);
      return;
    }

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
      EmitImmediate(Disp, 1, FK_Data_1, CurByte, OS, Fixups);
      return;
    }

    // Otherwise, emit the most general non-SIB encoding: [REG+disp32]
    EmitByte(ModRMByte(2, RegOpcodeField, BaseRegNo), CurByte, OS);
    EmitImmediate(Disp, 4, MCFixupKind(X86::reloc_signed_4byte), CurByte, OS,
                  Fixups);
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
  } else if (Disp.getImm() == 0 &&
             // Base reg can't be anything that ends up with '5' as the base
             // reg, it is the magic [*] nomenclature that indicates no base.
             BaseRegNo != N86::EBP) {
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
    EmitImmediate(Disp, 1, FK_Data_1, CurByte, OS, Fixups);
  else if (ForceDisp32 || Disp.getImm() != 0)
    EmitImmediate(Disp, 4, MCFixupKind(X86::reloc_signed_4byte), CurByte, OS,
                  Fixups);
}

/// EmitVEXOpcodePrefix - AVX instructions are encoded using a opcode prefix
/// called VEX.
void X86MCCodeEmitter::EmitVEXOpcodePrefix(uint64_t TSFlags, unsigned &CurByte,
                                           int MemOperand, const MCInst &MI,
                                           const TargetInstrDesc &Desc,
                                           raw_ostream &OS) const {
  bool HasVEX_4V = false;
  if ((TSFlags >> X86II::VEXShift) & X86II::VEX_4V)
    HasVEX_4V = true;

  // VEX_R: opcode externsion equivalent to REX.R in
  // 1's complement (inverted) form
  //
  //  1: Same as REX_R=0 (must be 1 in 32-bit mode)
  //  0: Same as REX_R=1 (64 bit mode only)
  //
  unsigned char VEX_R = 0x1;

  // VEX_X: equivalent to REX.X, only used when a
  // register is used for index in SIB Byte.
  //
  //  1: Same as REX.X=0 (must be 1 in 32-bit mode)
  //  0: Same as REX.X=1 (64-bit mode only)
  unsigned char VEX_X = 0x1;

  // VEX_B:
  //
  //  1: Same as REX_B=0 (ignored in 32-bit mode)
  //  0: Same as REX_B=1 (64 bit mode only)
  //
  unsigned char VEX_B = 0x1;

  // VEX_W: opcode specific (use like REX.W, or used for
  // opcode extension, or ignored, depending on the opcode byte)
  unsigned char VEX_W = 0;

  // VEX_5M (VEX m-mmmmm field):
  //
  //  0b00000: Reserved for future use
  //  0b00001: implied 0F leading opcode
  //  0b00010: implied 0F 38 leading opcode bytes
  //  0b00011: implied 0F 3A leading opcode bytes
  //  0b00100-0b11111: Reserved for future use
  //
  unsigned char VEX_5M = 0x1;

  // VEX_4V (VEX vvvv field): a register specifier
  // (in 1's complement form) or 1111 if unused.
  unsigned char VEX_4V = 0xf;

  // VEX_L (Vector Length):
  //
  //  0: scalar or 128-bit vector
  //  1: 256-bit vector
  //
  unsigned char VEX_L = 0;

  // VEX_PP: opcode extension providing equivalent
  // functionality of a SIMD prefix
  //
  //  0b00: None
  //  0b01: 66
  //  0b10: F3
  //  0b11: F2
  //
  unsigned char VEX_PP = 0;

  // Encode the operand size opcode prefix as needed.
  if (TSFlags & X86II::OpSize)
    VEX_PP = 0x01;

  if ((TSFlags >> X86II::VEXShift) & X86II::VEX_W)
    VEX_W = 1;

  if ((TSFlags >> X86II::VEXShift) & X86II::VEX_L)
    VEX_L = 1;

  switch (TSFlags & X86II::Op0Mask) {
  default: assert(0 && "Invalid prefix!");
  case X86II::T8:  // 0F 38
    VEX_5M = 0x2;
    break;
  case X86II::TA:  // 0F 3A
    VEX_5M = 0x3;
    break;
  case X86II::TF:  // F2 0F 38
    VEX_PP = 0x3;
    VEX_5M = 0x2;
    break;
  case X86II::XS:  // F3 0F
    VEX_PP = 0x2;
    break;
  case X86II::XD:  // F2 0F
    VEX_PP = 0x3;
    break;
  case X86II::A6:  // Bypass: Not used by VEX
  case X86II::A7:  // Bypass: Not used by VEX
  case X86II::TB:  // Bypass: Not used by VEX
  case 0:
    break;  // No prefix!
  }

  // Set the vector length to 256-bit if YMM0-YMM15 is used
  for (unsigned i = 0; i != MI.getNumOperands(); ++i) {
    if (!MI.getOperand(i).isReg())
      continue;
    unsigned SrcReg = MI.getOperand(i).getReg();
    if (SrcReg >= X86::YMM0 && SrcReg <= X86::YMM15)
      VEX_L = 1;
  }

  unsigned NumOps = MI.getNumOperands();
  unsigned CurOp = 0;
  bool IsDestMem = false;

  switch (TSFlags & X86II::FormMask) {
  case X86II::MRMInitReg: assert(0 && "FIXME: Remove this!");
  case X86II::MRMDestMem:
    IsDestMem = true;
    // The important info for the VEX prefix is never beyond the address
    // registers. Don't check beyond that.
    NumOps = CurOp = X86::AddrNumOperands;
  case X86II::MRM0m: case X86II::MRM1m:
  case X86II::MRM2m: case X86II::MRM3m:
  case X86II::MRM4m: case X86II::MRM5m:
  case X86II::MRM6m: case X86II::MRM7m:
  case X86II::MRMSrcMem:
  case X86II::MRMSrcReg:
    if (MI.getNumOperands() > CurOp && MI.getOperand(CurOp).isReg() &&
        X86InstrInfo::isX86_64ExtendedReg(MI.getOperand(CurOp).getReg()))
      VEX_R = 0x0;
    CurOp++;

    if (HasVEX_4V) {
      VEX_4V = getVEXRegisterEncoding(MI, IsDestMem ? CurOp-1 : CurOp);
      CurOp++;
    }

    // To only check operands before the memory address ones, start
    // the search from the beginning
    if (IsDestMem)
      CurOp = 0;

    // If the last register should be encoded in the immediate field
    // do not use any bit from VEX prefix to this register, ignore it
    if ((TSFlags >> X86II::VEXShift) & X86II::VEX_I8IMM)
      NumOps--;

    for (; CurOp != NumOps; ++CurOp) {
      const MCOperand &MO = MI.getOperand(CurOp);
      if (MO.isReg() && X86InstrInfo::isX86_64ExtendedReg(MO.getReg()))
        VEX_B = 0x0;
      if (!VEX_B && MO.isReg() &&
          ((TSFlags & X86II::FormMask) == X86II::MRMSrcMem) &&
          X86InstrInfo::isX86_64ExtendedReg(MO.getReg()))
        VEX_X = 0x0;
    }
    break;
  default: // MRMDestReg, MRM0r-MRM7r, RawFrm
    if (!MI.getNumOperands())
      break;

    if (MI.getOperand(CurOp).isReg() &&
        X86InstrInfo::isX86_64ExtendedReg(MI.getOperand(CurOp).getReg()))
      VEX_B = 0;

    if (HasVEX_4V)
      VEX_4V = getVEXRegisterEncoding(MI, CurOp);

    CurOp++;
    for (; CurOp != NumOps; ++CurOp) {
      const MCOperand &MO = MI.getOperand(CurOp);
      if (MO.isReg() && !HasVEX_4V &&
          X86InstrInfo::isX86_64ExtendedReg(MO.getReg()))
        VEX_R = 0x0;
    }
    break;
  }

  // Emit segment override opcode prefix as needed.
  EmitSegmentOverridePrefix(TSFlags, CurByte, MemOperand, MI, OS);

  // VEX opcode prefix can have 2 or 3 bytes
  //
  //  3 bytes:
  //    +-----+ +--------------+ +-------------------+
  //    | C4h | | RXB | m-mmmm | | W | vvvv | L | pp |
  //    +-----+ +--------------+ +-------------------+
  //  2 bytes:
  //    +-----+ +-------------------+
  //    | C5h | | R | vvvv | L | pp |
  //    +-----+ +-------------------+
  //
  unsigned char LastByte = VEX_PP | (VEX_L << 2) | (VEX_4V << 3);

  if (VEX_B && VEX_X && !VEX_W && (VEX_5M == 1)) { // 2 byte VEX prefix
    EmitByte(0xC5, CurByte, OS);
    EmitByte(LastByte | (VEX_R << 7), CurByte, OS);
    return;
  }

  // 3 byte VEX prefix
  EmitByte(0xC4, CurByte, OS);
  EmitByte(VEX_R << 7 | VEX_X << 6 | VEX_B << 5 | VEX_5M, CurByte, OS);
  EmitByte(LastByte | (VEX_W << 7), CurByte, OS);
}

/// DetermineREXPrefix - Determine if the MCInst has to be encoded with a X86-64
/// REX prefix which specifies 1) 64-bit instructions, 2) non-default operand
/// size, and 3) use of X86-64 extended registers.
static unsigned DetermineREXPrefix(const MCInst &MI, uint64_t TSFlags,
                                   const TargetInstrDesc &Desc) {
  unsigned REX = 0;
  if (TSFlags & X86II::REX_W)
    REX |= 1 << 3; // set REX.W

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
    REX |= 0x40; // REX fixed encoding prefix
    break;
  }

  switch (TSFlags & X86II::FormMask) {
  case X86II::MRMInitReg: assert(0 && "FIXME: Remove this!");
  case X86II::MRMSrcReg:
    if (MI.getOperand(0).isReg() &&
        X86InstrInfo::isX86_64ExtendedReg(MI.getOperand(0).getReg()))
      REX |= 1 << 2; // set REX.R
    i = isTwoAddr ? 2 : 1;
    for (; i != NumOps; ++i) {
      const MCOperand &MO = MI.getOperand(i);
      if (MO.isReg() && X86InstrInfo::isX86_64ExtendedReg(MO.getReg()))
        REX |= 1 << 0; // set REX.B
    }
    break;
  case X86II::MRMSrcMem: {
    if (MI.getOperand(0).isReg() &&
        X86InstrInfo::isX86_64ExtendedReg(MI.getOperand(0).getReg()))
      REX |= 1 << 2; // set REX.R
    unsigned Bit = 0;
    i = isTwoAddr ? 2 : 1;
    for (; i != NumOps; ++i) {
      const MCOperand &MO = MI.getOperand(i);
      if (MO.isReg()) {
        if (X86InstrInfo::isX86_64ExtendedReg(MO.getReg()))
          REX |= 1 << Bit; // set REX.B (Bit=0) and REX.X (Bit=1)
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
    unsigned e = (isTwoAddr ? X86::AddrNumOperands+1 : X86::AddrNumOperands);
    i = isTwoAddr ? 1 : 0;
    if (NumOps > e && MI.getOperand(e).isReg() &&
        X86InstrInfo::isX86_64ExtendedReg(MI.getOperand(e).getReg()))
      REX |= 1 << 2; // set REX.R
    unsigned Bit = 0;
    for (; i != e; ++i) {
      const MCOperand &MO = MI.getOperand(i);
      if (MO.isReg()) {
        if (X86InstrInfo::isX86_64ExtendedReg(MO.getReg()))
          REX |= 1 << Bit; // REX.B (Bit=0) and REX.X (Bit=1)
        Bit++;
      }
    }
    break;
  }
  default:
    if (MI.getOperand(0).isReg() &&
        X86InstrInfo::isX86_64ExtendedReg(MI.getOperand(0).getReg()))
      REX |= 1 << 0; // set REX.B
    i = isTwoAddr ? 2 : 1;
    for (unsigned e = NumOps; i != e; ++i) {
      const MCOperand &MO = MI.getOperand(i);
      if (MO.isReg() && X86InstrInfo::isX86_64ExtendedReg(MO.getReg()))
        REX |= 1 << 2; // set REX.R
    }
    break;
  }
  return REX;
}

/// EmitSegmentOverridePrefix - Emit segment override opcode prefix as needed
void X86MCCodeEmitter::EmitSegmentOverridePrefix(uint64_t TSFlags,
                                        unsigned &CurByte, int MemOperand,
                                        const MCInst &MI,
                                        raw_ostream &OS) const {
  switch (TSFlags & X86II::SegOvrMask) {
  default: assert(0 && "Invalid segment!");
  case 0:
    // No segment override, check for explicit one on memory operand.
    if (MemOperand != -1) {   // If the instruction has a memory operand.
      switch (MI.getOperand(MemOperand+X86::AddrSegmentReg).getReg()) {
      default: assert(0 && "Unknown segment register!");
      case 0: break;
      case X86::CS: EmitByte(0x2E, CurByte, OS); break;
      case X86::SS: EmitByte(0x36, CurByte, OS); break;
      case X86::DS: EmitByte(0x3E, CurByte, OS); break;
      case X86::ES: EmitByte(0x26, CurByte, OS); break;
      case X86::FS: EmitByte(0x64, CurByte, OS); break;
      case X86::GS: EmitByte(0x65, CurByte, OS); break;
      }
    }
    break;
  case X86II::FS:
    EmitByte(0x64, CurByte, OS);
    break;
  case X86II::GS:
    EmitByte(0x65, CurByte, OS);
    break;
  }
}

/// EmitOpcodePrefix - Emit all instruction prefixes prior to the opcode.
///
/// MemOperand is the operand # of the start of a memory operand if present.  If
/// Not present, it is -1.
void X86MCCodeEmitter::EmitOpcodePrefix(uint64_t TSFlags, unsigned &CurByte,
                                        int MemOperand, const MCInst &MI,
                                        const TargetInstrDesc &Desc,
                                        raw_ostream &OS) const {

  // Emit the lock opcode prefix as needed.
  if (TSFlags & X86II::LOCK)
    EmitByte(0xF0, CurByte, OS);

  // Emit segment override opcode prefix as needed.
  EmitSegmentOverridePrefix(TSFlags, CurByte, MemOperand, MI, OS);

  // Emit the repeat opcode prefix as needed.
  if ((TSFlags & X86II::Op0Mask) == X86II::REP)
    EmitByte(0xF3, CurByte, OS);

  // Emit the address size opcode prefix as needed.
  if ((TSFlags & X86II::AdSize) ||
      (MemOperand != -1 && Is64BitMode && Is32BitMemOperand(MI, MemOperand)))
    EmitByte(0x67, CurByte, OS);
  
  // Emit the operand size opcode prefix as needed.
  if (TSFlags & X86II::OpSize)
    EmitByte(0x66, CurByte, OS);

  bool Need0FPrefix = false;
  switch (TSFlags & X86II::Op0Mask) {
  default: assert(0 && "Invalid prefix!");
  case 0: break;  // No prefix!
  case X86II::REP: break; // already handled.
  case X86II::TB:  // Two-byte opcode prefix
  case X86II::T8:  // 0F 38
  case X86II::TA:  // 0F 3A
  case X86II::A6:  // 0F A6
  case X86II::A7:  // 0F A7
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
  case X86II::A6:    // 0F A6
    EmitByte(0xA6, CurByte, OS);
    break;
  case X86II::A7:    // 0F A7
    EmitByte(0xA7, CurByte, OS);
    break;
  }
}

void X86MCCodeEmitter::
EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                  SmallVectorImpl<MCFixup> &Fixups) const {
  unsigned Opcode = MI.getOpcode();
  const TargetInstrDesc &Desc = TII.get(Opcode);
  uint64_t TSFlags = Desc.TSFlags;

  // Pseudo instructions don't get encoded.
  if ((TSFlags & X86II::FormMask) == X86II::Pseudo)
    return;

  // If this is a two-address instruction, skip one of the register operands.
  // FIXME: This should be handled during MCInst lowering.
  unsigned NumOps = Desc.getNumOperands();
  unsigned CurOp = 0;
  if (NumOps > 1 && Desc.getOperandConstraint(1, TOI::TIED_TO) != -1)
    ++CurOp;
  else if (NumOps > 2 && Desc.getOperandConstraint(NumOps-1, TOI::TIED_TO)== 0)
    // Skip the last source operand that is tied_to the dest reg. e.g. LXADD32
    --NumOps;

  // Keep track of the current byte being emitted.
  unsigned CurByte = 0;

  // Is this instruction encoded using the AVX VEX prefix?
  bool HasVEXPrefix = false;

  // It uses the VEX.VVVV field?
  bool HasVEX_4V = false;

  if ((TSFlags >> X86II::VEXShift) & X86II::VEX)
    HasVEXPrefix = true;
  if ((TSFlags >> X86II::VEXShift) & X86II::VEX_4V)
    HasVEX_4V = true;

  
  // Determine where the memory operand starts, if present.
  int MemoryOperand = X86II::getMemoryOperandNo(TSFlags);
  if (MemoryOperand != -1) MemoryOperand += CurOp;

  if (!HasVEXPrefix)
    EmitOpcodePrefix(TSFlags, CurByte, MemoryOperand, MI, Desc, OS);
  else
    EmitVEXOpcodePrefix(TSFlags, CurByte, MemoryOperand, MI, Desc, OS);

  
  unsigned char BaseOpcode = X86II::getBaseOpcodeFor(TSFlags);
  
  if ((TSFlags >> X86II::VEXShift) & X86II::Has3DNow0F0FOpcode)
    BaseOpcode = 0x0F;   // Weird 3DNow! encoding.
  
  unsigned SrcRegNum = 0;
  switch (TSFlags & X86II::FormMask) {
  case X86II::MRMInitReg:
    assert(0 && "FIXME: Remove this form when the JIT moves to MCCodeEmitter!");
  default: errs() << "FORM: " << (TSFlags & X86II::FormMask) << "\n";
    assert(0 && "Unknown FormMask value in X86MCCodeEmitter!");
  case X86II::Pseudo:
    assert(0 && "Pseudo instruction shouldn't be emitted");
  case X86II::RawFrm:
    EmitByte(BaseOpcode, CurByte, OS);
    break;
      
  case X86II::RawFrmImm8:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitImmediate(MI.getOperand(CurOp++),
                  X86II::getSizeOfImm(TSFlags), getImmFixupKind(TSFlags),
                  CurByte, OS, Fixups);
    EmitImmediate(MI.getOperand(CurOp++), 1, FK_Data_1, CurByte, OS, Fixups);
    break;
  case X86II::RawFrmImm16:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitImmediate(MI.getOperand(CurOp++),
                  X86II::getSizeOfImm(TSFlags), getImmFixupKind(TSFlags),
                  CurByte, OS, Fixups);
    EmitImmediate(MI.getOperand(CurOp++), 2, FK_Data_2, CurByte, OS, Fixups);
    break;

  case X86II::AddRegFrm:
    EmitByte(BaseOpcode + GetX86RegNum(MI.getOperand(CurOp++)), CurByte, OS);
    break;

  case X86II::MRMDestReg:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitRegModRMByte(MI.getOperand(CurOp),
                     GetX86RegNum(MI.getOperand(CurOp+1)), CurByte, OS);
    CurOp += 2;
    break;

  case X86II::MRMDestMem:
    EmitByte(BaseOpcode, CurByte, OS);
    SrcRegNum = CurOp + X86::AddrNumOperands;

    if (HasVEX_4V) // Skip 1st src (which is encoded in VEX_VVVV)
      SrcRegNum++;

    EmitMemModRMByte(MI, CurOp,
                     GetX86RegNum(MI.getOperand(SrcRegNum)),
                     TSFlags, CurByte, OS, Fixups);
    CurOp = SrcRegNum + 1;
    break;

  case X86II::MRMSrcReg:
    EmitByte(BaseOpcode, CurByte, OS);
    SrcRegNum = CurOp + 1;

    if (HasVEX_4V) // Skip 1st src (which is encoded in VEX_VVVV)
      SrcRegNum++;

    EmitRegModRMByte(MI.getOperand(SrcRegNum),
                     GetX86RegNum(MI.getOperand(CurOp)), CurByte, OS);
    CurOp = SrcRegNum + 1;
    break;

  case X86II::MRMSrcMem: {
    int AddrOperands = X86::AddrNumOperands;
    unsigned FirstMemOp = CurOp+1;
    if (HasVEX_4V) {
      ++AddrOperands;
      ++FirstMemOp;  // Skip the register source (which is encoded in VEX_VVVV).
    }

    EmitByte(BaseOpcode, CurByte, OS);

    EmitMemModRMByte(MI, FirstMemOp, GetX86RegNum(MI.getOperand(CurOp)),
                     TSFlags, CurByte, OS, Fixups);
    CurOp += AddrOperands + 1;
    break;
  }

  case X86II::MRM0r: case X86II::MRM1r:
  case X86II::MRM2r: case X86II::MRM3r:
  case X86II::MRM4r: case X86II::MRM5r:
  case X86II::MRM6r: case X86II::MRM7r:
    if (HasVEX_4V) // Skip the register dst (which is encoded in VEX_VVVV).
      CurOp++;
    EmitByte(BaseOpcode, CurByte, OS);
    EmitRegModRMByte(MI.getOperand(CurOp++),
                     (TSFlags & X86II::FormMask)-X86II::MRM0r,
                     CurByte, OS);
    break;
  case X86II::MRM0m: case X86II::MRM1m:
  case X86II::MRM2m: case X86II::MRM3m:
  case X86II::MRM4m: case X86II::MRM5m:
  case X86II::MRM6m: case X86II::MRM7m:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitMemModRMByte(MI, CurOp, (TSFlags & X86II::FormMask)-X86II::MRM0m,
                     TSFlags, CurByte, OS, Fixups);
    CurOp += X86::AddrNumOperands;
    break;
  case X86II::MRM_C1:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitByte(0xC1, CurByte, OS);
    break;
  case X86II::MRM_C2:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitByte(0xC2, CurByte, OS);
    break;
  case X86II::MRM_C3:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitByte(0xC3, CurByte, OS);
    break;
  case X86II::MRM_C4:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitByte(0xC4, CurByte, OS);
    break;
  case X86II::MRM_C8:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitByte(0xC8, CurByte, OS);
    break;
  case X86II::MRM_C9:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitByte(0xC9, CurByte, OS);
    break;
  case X86II::MRM_E8:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitByte(0xE8, CurByte, OS);
    break;
  case X86II::MRM_F0:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitByte(0xF0, CurByte, OS);
    break;
  case X86II::MRM_F8:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitByte(0xF8, CurByte, OS);
    break;
  case X86II::MRM_F9:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitByte(0xF9, CurByte, OS);
    break;
  case X86II::MRM_D0:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitByte(0xD0, CurByte, OS);
    break;
  case X86II::MRM_D1:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitByte(0xD1, CurByte, OS);
    break;
  }

  // If there is a remaining operand, it must be a trailing immediate.  Emit it
  // according to the right size for the instruction.
  if (CurOp != NumOps) {
    // The last source register of a 4 operand instruction in AVX is encoded
    // in bits[7:4] of a immediate byte, and bits[3:0] are ignored.
    if ((TSFlags >> X86II::VEXShift) & X86II::VEX_I8IMM) {
      const MCOperand &MO = MI.getOperand(CurOp++);
      bool IsExtReg =
        X86InstrInfo::isX86_64ExtendedReg(MO.getReg());
      unsigned RegNum = (IsExtReg ? (1 << 7) : 0);
      RegNum |= GetX86RegNum(MO) << 4;
      EmitImmediate(MCOperand::CreateImm(RegNum), 1, FK_Data_1, CurByte, OS,
                    Fixups);
    } else {
      unsigned FixupKind;
      // FIXME: Is there a better way to know that we need a signed relocation?
      if (MI.getOpcode() == X86::MOV64ri32 ||
          MI.getOpcode() == X86::MOV64mi32 ||
          MI.getOpcode() == X86::PUSH64i32)
        FixupKind = X86::reloc_signed_4byte;
      else
        FixupKind = getImmFixupKind(TSFlags);
      EmitImmediate(MI.getOperand(CurOp++),
                    X86II::getSizeOfImm(TSFlags), MCFixupKind(FixupKind),
                    CurByte, OS, Fixups);
    }
  }

  if ((TSFlags >> X86II::VEXShift) & X86II::Has3DNow0F0FOpcode)
    EmitByte(X86II::getBaseOpcodeFor(TSFlags), CurByte, OS);
  

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
