//===-- X86MCCodeEmitter.cpp - Convert X86 code to machine code -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the X86MCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "MCTargetDesc/X86FixupKinds.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>

using namespace llvm;

#define DEBUG_TYPE "mccodeemitter"

namespace {

class X86MCCodeEmitter : public MCCodeEmitter {
  const MCInstrInfo &MCII;
  MCContext &Ctx;

public:
  X86MCCodeEmitter(const MCInstrInfo &mcii, MCContext &ctx)
      : MCII(mcii), Ctx(ctx) {}
  X86MCCodeEmitter(const X86MCCodeEmitter &) = delete;
  X86MCCodeEmitter &operator=(const X86MCCodeEmitter &) = delete;
  ~X86MCCodeEmitter() override = default;

  void emitPrefix(const MCInst &MI, raw_ostream &OS,
                  const MCSubtargetInfo &STI) const override;

  void encodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

private:
  unsigned getX86RegNum(const MCOperand &MO) const;

  unsigned getX86RegEncoding(const MCInst &MI, unsigned OpNum) const;

  /// \param MI a single low-level machine instruction.
  /// \param OpNum the operand #.
  /// \returns true if the OpNumth operand of MI  require a bit to be set in
  /// REX prefix.
  bool isREXExtendedReg(const MCInst &MI, unsigned OpNum) const;

  void emitImmediate(const MCOperand &Disp, SMLoc Loc, unsigned ImmSize,
                     MCFixupKind FixupKind, uint64_t StartByte, raw_ostream &OS,
                     SmallVectorImpl<MCFixup> &Fixups, int ImmOffset = 0) const;

  void emitRegModRMByte(const MCOperand &ModRMReg, unsigned RegOpcodeFld,
                        raw_ostream &OS) const;

  void emitSIBByte(unsigned SS, unsigned Index, unsigned Base,
                   raw_ostream &OS) const;

  void emitMemModRMByte(const MCInst &MI, unsigned Op, unsigned RegOpcodeField,
                        uint64_t TSFlags, bool HasREX, uint64_t StartByte,
                        raw_ostream &OS, SmallVectorImpl<MCFixup> &Fixups,
                        const MCSubtargetInfo &STI,
                        bool ForceSIB = false) const;

  bool emitPrefixImpl(unsigned &CurOp, const MCInst &MI,
                      const MCSubtargetInfo &STI, raw_ostream &OS) const;

  void emitVEXOpcodePrefix(int MemOperand, const MCInst &MI,
                           raw_ostream &OS) const;

  void emitSegmentOverridePrefix(unsigned SegOperand, const MCInst &MI,
                                 raw_ostream &OS) const;

  bool emitOpcodePrefix(int MemOperand, const MCInst &MI,
                        const MCSubtargetInfo &STI, raw_ostream &OS) const;

  bool emitREXPrefix(int MemOperand, const MCInst &MI,
                     const MCSubtargetInfo &STI, raw_ostream &OS) const;
};

} // end anonymous namespace

static uint8_t modRMByte(unsigned Mod, unsigned RegOpcode, unsigned RM) {
  assert(Mod < 4 && RegOpcode < 8 && RM < 8 && "ModRM Fields out of range!");
  return RM | (RegOpcode << 3) | (Mod << 6);
}

static void emitByte(uint8_t C, raw_ostream &OS) { OS << static_cast<char>(C); }

static void emitConstant(uint64_t Val, unsigned Size, raw_ostream &OS) {
  // Output the constant in little endian byte order.
  for (unsigned i = 0; i != Size; ++i) {
    emitByte(Val & 255, OS);
    Val >>= 8;
  }
}

/// Determine if this immediate can fit in a disp8 or a compressed disp8 for
/// EVEX instructions. \p will be set to the value to pass to the ImmOffset
/// parameter of emitImmediate.
static bool isDispOrCDisp8(uint64_t TSFlags, int Value, int &ImmOffset) {
  bool HasEVEX = (TSFlags & X86II::EncodingMask) == X86II::EVEX;

  int CD8_Scale =
      (TSFlags & X86II::CD8_Scale_Mask) >> X86II::CD8_Scale_Shift;
  if (!HasEVEX || CD8_Scale == 0)
    return isInt<8>(Value);

  assert(isPowerOf2_32(CD8_Scale) && "Unexpected CD8 scale!");
  if (Value & (CD8_Scale - 1)) // Unaligned offset
    return false;

  int CDisp8 = Value / CD8_Scale;
  if (!isInt<8>(CDisp8))
    return false;

  // ImmOffset will be added to Value in emitImmediate leaving just CDisp8.
  ImmOffset = CDisp8 - Value;
  return true;
}

/// \returns the appropriate fixup kind to use for an immediate in an
/// instruction with the specified TSFlags.
static MCFixupKind getImmFixupKind(uint64_t TSFlags) {
  unsigned Size = X86II::getSizeOfImm(TSFlags);
  bool isPCRel = X86II::isImmPCRel(TSFlags);

  if (X86II::isImmSigned(TSFlags)) {
    switch (Size) {
    default:
      llvm_unreachable("Unsupported signed fixup size!");
    case 4:
      return MCFixupKind(X86::reloc_signed_4byte);
    }
  }
  return MCFixup::getKindForSize(Size, isPCRel);
}

enum GlobalOffsetTableExprKind { GOT_None, GOT_Normal, GOT_SymDiff };

/// Check if this expression starts with  _GLOBAL_OFFSET_TABLE_ and if it is
/// of the form _GLOBAL_OFFSET_TABLE_-symbol. This is needed to support PIC on
/// ELF i386 as _GLOBAL_OFFSET_TABLE_ is magical. We check only simple case that
/// are know to be used: _GLOBAL_OFFSET_TABLE_ by itself or at the start of a
/// binary expression.
static GlobalOffsetTableExprKind
startsWithGlobalOffsetTable(const MCExpr *Expr) {
  const MCExpr *RHS = nullptr;
  if (Expr->getKind() == MCExpr::Binary) {
    const MCBinaryExpr *BE = static_cast<const MCBinaryExpr *>(Expr);
    Expr = BE->getLHS();
    RHS = BE->getRHS();
  }

  if (Expr->getKind() != MCExpr::SymbolRef)
    return GOT_None;

  const MCSymbolRefExpr *Ref = static_cast<const MCSymbolRefExpr *>(Expr);
  const MCSymbol &S = Ref->getSymbol();
  if (S.getName() != "_GLOBAL_OFFSET_TABLE_")
    return GOT_None;
  if (RHS && RHS->getKind() == MCExpr::SymbolRef)
    return GOT_SymDiff;
  return GOT_Normal;
}

static bool hasSecRelSymbolRef(const MCExpr *Expr) {
  if (Expr->getKind() == MCExpr::SymbolRef) {
    const MCSymbolRefExpr *Ref = static_cast<const MCSymbolRefExpr *>(Expr);
    return Ref->getKind() == MCSymbolRefExpr::VK_SECREL;
  }
  return false;
}

static bool isPCRel32Branch(const MCInst &MI, const MCInstrInfo &MCII) {
  unsigned Opcode = MI.getOpcode();
  const MCInstrDesc &Desc = MCII.get(Opcode);
  if ((Opcode != X86::CALL64pcrel32 && Opcode != X86::JMP_4 &&
       Opcode != X86::JCC_4) ||
      getImmFixupKind(Desc.TSFlags) != FK_PCRel_4)
    return false;

  unsigned CurOp = X86II::getOperandBias(Desc);
  const MCOperand &Op = MI.getOperand(CurOp);
  if (!Op.isExpr())
    return false;

  const MCSymbolRefExpr *Ref = dyn_cast<MCSymbolRefExpr>(Op.getExpr());
  return Ref && Ref->getKind() == MCSymbolRefExpr::VK_None;
}

unsigned X86MCCodeEmitter::getX86RegNum(const MCOperand &MO) const {
  return Ctx.getRegisterInfo()->getEncodingValue(MO.getReg()) & 0x7;
}

unsigned X86MCCodeEmitter::getX86RegEncoding(const MCInst &MI,
                                             unsigned OpNum) const {
  return Ctx.getRegisterInfo()->getEncodingValue(MI.getOperand(OpNum).getReg());
}

/// \param MI a single low-level machine instruction.
/// \param OpNum the operand #.
/// \returns true if the OpNumth operand of MI  require a bit to be set in
/// REX prefix.
bool X86MCCodeEmitter::isREXExtendedReg(const MCInst &MI,
                                        unsigned OpNum) const {
  return (getX86RegEncoding(MI, OpNum) >> 3) & 1;
}

void X86MCCodeEmitter::emitImmediate(const MCOperand &DispOp, SMLoc Loc,
                                     unsigned Size, MCFixupKind FixupKind,
                                     uint64_t StartByte, raw_ostream &OS,
                                     SmallVectorImpl<MCFixup> &Fixups,
                                     int ImmOffset) const {
  const MCExpr *Expr = nullptr;
  if (DispOp.isImm()) {
    // If this is a simple integer displacement that doesn't require a
    // relocation, emit it now.
    if (FixupKind != FK_PCRel_1 && FixupKind != FK_PCRel_2 &&
        FixupKind != FK_PCRel_4) {
      emitConstant(DispOp.getImm() + ImmOffset, Size, OS);
      return;
    }
    Expr = MCConstantExpr::create(DispOp.getImm(), Ctx);
  } else {
    Expr = DispOp.getExpr();
  }

  // If we have an immoffset, add it to the expression.
  if ((FixupKind == FK_Data_4 || FixupKind == FK_Data_8 ||
       FixupKind == MCFixupKind(X86::reloc_signed_4byte))) {
    GlobalOffsetTableExprKind Kind = startsWithGlobalOffsetTable(Expr);
    if (Kind != GOT_None) {
      assert(ImmOffset == 0);

      if (Size == 8) {
        FixupKind = MCFixupKind(X86::reloc_global_offset_table8);
      } else {
        assert(Size == 4);
        FixupKind = MCFixupKind(X86::reloc_global_offset_table);
      }

      if (Kind == GOT_Normal)
        ImmOffset = static_cast<int>(OS.tell() - StartByte);
    } else if (Expr->getKind() == MCExpr::SymbolRef) {
      if (hasSecRelSymbolRef(Expr)) {
        FixupKind = MCFixupKind(FK_SecRel_4);
      }
    } else if (Expr->getKind() == MCExpr::Binary) {
      const MCBinaryExpr *Bin = static_cast<const MCBinaryExpr *>(Expr);
      if (hasSecRelSymbolRef(Bin->getLHS()) ||
          hasSecRelSymbolRef(Bin->getRHS())) {
        FixupKind = MCFixupKind(FK_SecRel_4);
      }
    }
  }

  // If the fixup is pc-relative, we need to bias the value to be relative to
  // the start of the field, not the end of the field.
  if (FixupKind == FK_PCRel_4 ||
      FixupKind == MCFixupKind(X86::reloc_riprel_4byte) ||
      FixupKind == MCFixupKind(X86::reloc_riprel_4byte_movq_load) ||
      FixupKind == MCFixupKind(X86::reloc_riprel_4byte_relax) ||
      FixupKind == MCFixupKind(X86::reloc_riprel_4byte_relax_rex) ||
      FixupKind == MCFixupKind(X86::reloc_branch_4byte_pcrel)) {
    ImmOffset -= 4;
    // If this is a pc-relative load off _GLOBAL_OFFSET_TABLE_:
    // leaq _GLOBAL_OFFSET_TABLE_(%rip), %r15
    // this needs to be a GOTPC32 relocation.
    if (startsWithGlobalOffsetTable(Expr) != GOT_None)
      FixupKind = MCFixupKind(X86::reloc_global_offset_table);
  }
  if (FixupKind == FK_PCRel_2)
    ImmOffset -= 2;
  if (FixupKind == FK_PCRel_1)
    ImmOffset -= 1;

  if (ImmOffset)
    Expr = MCBinaryExpr::createAdd(Expr, MCConstantExpr::create(ImmOffset, Ctx),
                                   Ctx);

  // Emit a symbolic constant as a fixup and 4 zeros.
  Fixups.push_back(MCFixup::create(static_cast<uint32_t>(OS.tell() - StartByte),
                                   Expr, FixupKind, Loc));
  emitConstant(0, Size, OS);
}

void X86MCCodeEmitter::emitRegModRMByte(const MCOperand &ModRMReg,
                                        unsigned RegOpcodeFld,
                                        raw_ostream &OS) const {
  emitByte(modRMByte(3, RegOpcodeFld, getX86RegNum(ModRMReg)), OS);
}

void X86MCCodeEmitter::emitSIBByte(unsigned SS, unsigned Index, unsigned Base,
                                   raw_ostream &OS) const {
  // SIB byte is in the same format as the modRMByte.
  emitByte(modRMByte(SS, Index, Base), OS);
}

void X86MCCodeEmitter::emitMemModRMByte(const MCInst &MI, unsigned Op,
                                        unsigned RegOpcodeField,
                                        uint64_t TSFlags, bool HasREX,
                                        uint64_t StartByte, raw_ostream &OS,
                                        SmallVectorImpl<MCFixup> &Fixups,
                                        const MCSubtargetInfo &STI,
                                        bool ForceSIB) const {
  const MCOperand &Disp = MI.getOperand(Op + X86::AddrDisp);
  const MCOperand &Base = MI.getOperand(Op + X86::AddrBaseReg);
  const MCOperand &Scale = MI.getOperand(Op + X86::AddrScaleAmt);
  const MCOperand &IndexReg = MI.getOperand(Op + X86::AddrIndexReg);
  unsigned BaseReg = Base.getReg();

  // Handle %rip relative addressing.
  if (BaseReg == X86::RIP ||
      BaseReg == X86::EIP) { // [disp32+rIP] in X86-64 mode
    assert(STI.hasFeature(X86::Is64Bit) &&
           "Rip-relative addressing requires 64-bit mode");
    assert(IndexReg.getReg() == 0 && !ForceSIB &&
           "Invalid rip-relative address");
    emitByte(modRMByte(0, RegOpcodeField, 5), OS);

    unsigned Opcode = MI.getOpcode();
    unsigned FixupKind = [&]() {
      // Enable relaxed relocation only for a MCSymbolRefExpr.  We cannot use a
      // relaxed relocation if an offset is present (e.g. x@GOTPCREL+4).
      if (!(Disp.isExpr() && isa<MCSymbolRefExpr>(Disp.getExpr())))
        return X86::reloc_riprel_4byte;

      // Certain loads for GOT references can be relocated against the symbol
      // directly if the symbol ends up in the same linkage unit.
      switch (Opcode) {
      default:
        return X86::reloc_riprel_4byte;
      case X86::MOV64rm:
        // movq loads is a subset of reloc_riprel_4byte_relax_rex. It is a
        // special case because COFF and Mach-O don't support ELF's more
        // flexible R_X86_64_REX_GOTPCRELX relaxation.
        assert(HasREX);
        return X86::reloc_riprel_4byte_movq_load;
      case X86::ADC32rm:
      case X86::ADD32rm:
      case X86::AND32rm:
      case X86::CMP32rm:
      case X86::MOV32rm:
      case X86::OR32rm:
      case X86::SBB32rm:
      case X86::SUB32rm:
      case X86::TEST32mr:
      case X86::XOR32rm:
      case X86::CALL64m:
      case X86::JMP64m:
      case X86::TAILJMPm64:
      case X86::TEST64mr:
      case X86::ADC64rm:
      case X86::ADD64rm:
      case X86::AND64rm:
      case X86::CMP64rm:
      case X86::OR64rm:
      case X86::SBB64rm:
      case X86::SUB64rm:
      case X86::XOR64rm:
        return HasREX ? X86::reloc_riprel_4byte_relax_rex
                      : X86::reloc_riprel_4byte_relax;
      }
    }();

    // rip-relative addressing is actually relative to the *next* instruction.
    // Since an immediate can follow the mod/rm byte for an instruction, this
    // means that we need to bias the displacement field of the instruction with
    // the size of the immediate field. If we have this case, add it into the
    // expression to emit.
    // Note: rip-relative addressing using immediate displacement values should
    // not be adjusted, assuming it was the user's intent.
    int ImmSize = !Disp.isImm() && X86II::hasImm(TSFlags)
                      ? X86II::getSizeOfImm(TSFlags)
                      : 0;

    emitImmediate(Disp, MI.getLoc(), 4, MCFixupKind(FixupKind), StartByte, OS,
                  Fixups, -ImmSize);
    return;
  }

  unsigned BaseRegNo = BaseReg ? getX86RegNum(Base) : -1U;

  // 16-bit addressing forms of the ModR/M byte have a different encoding for
  // the R/M field and are far more limited in which registers can be used.
  if (X86_MC::is16BitMemOperand(MI, Op, STI)) {
    if (BaseReg) {
      // For 32-bit addressing, the row and column values in Table 2-2 are
      // basically the same. It's AX/CX/DX/BX/SP/BP/SI/DI in that order, with
      // some special cases. And getX86RegNum reflects that numbering.
      // For 16-bit addressing it's more fun, as shown in the SDM Vol 2A,
      // Table 2-1 "16-Bit Addressing Forms with the ModR/M byte". We can only
      // use SI/DI/BP/BX, which have "row" values 4-7 in no particular order,
      // while values 0-3 indicate the allowed combinations (base+index) of
      // those: 0 for BX+SI, 1 for BX+DI, 2 for BP+SI, 3 for BP+DI.
      //
      // R16Table[] is a lookup from the normal RegNo, to the row values from
      // Table 2-1 for 16-bit addressing modes. Where zero means disallowed.
      static const unsigned R16Table[] = {0, 0, 0, 7, 0, 6, 4, 5};
      unsigned RMfield = R16Table[BaseRegNo];

      assert(RMfield && "invalid 16-bit base register");

      if (IndexReg.getReg()) {
        unsigned IndexReg16 = R16Table[getX86RegNum(IndexReg)];

        assert(IndexReg16 && "invalid 16-bit index register");
        // We must have one of SI/DI (4,5), and one of BP/BX (6,7).
        assert(((IndexReg16 ^ RMfield) & 2) &&
               "invalid 16-bit base/index register combination");
        assert(Scale.getImm() == 1 &&
               "invalid scale for 16-bit memory reference");

        // Allow base/index to appear in either order (although GAS doesn't).
        if (IndexReg16 & 2)
          RMfield = (RMfield & 1) | ((7 - IndexReg16) << 1);
        else
          RMfield = (IndexReg16 & 1) | ((7 - RMfield) << 1);
      }

      if (Disp.isImm() && isInt<8>(Disp.getImm())) {
        if (Disp.getImm() == 0 && RMfield != 6) {
          // There is no displacement; just the register.
          emitByte(modRMByte(0, RegOpcodeField, RMfield), OS);
          return;
        }
        // Use the [REG]+disp8 form, including for [BP] which cannot be encoded.
        emitByte(modRMByte(1, RegOpcodeField, RMfield), OS);
        emitImmediate(Disp, MI.getLoc(), 1, FK_Data_1, StartByte, OS, Fixups);
        return;
      }
      // This is the [REG]+disp16 case.
      emitByte(modRMByte(2, RegOpcodeField, RMfield), OS);
    } else {
      assert(IndexReg.getReg() == 0 && "Unexpected index register!");
      // There is no BaseReg; this is the plain [disp16] case.
      emitByte(modRMByte(0, RegOpcodeField, 6), OS);
    }

    // Emit 16-bit displacement for plain disp16 or [REG]+disp16 cases.
    emitImmediate(Disp, MI.getLoc(), 2, FK_Data_2, StartByte, OS, Fixups);
    return;
  }

  // Check for presence of {disp8} or {disp32} pseudo prefixes.
  bool UseDisp8 = MI.getFlags() & X86::IP_USE_DISP8;
  bool UseDisp32 = MI.getFlags() & X86::IP_USE_DISP32;

  // We only allow no displacement if no pseudo prefix is present.
  bool AllowNoDisp = !UseDisp8 && !UseDisp32;
  // Disp8 is allowed unless the {disp32} prefix is present.
  bool AllowDisp8 = !UseDisp32;

  // Determine whether a SIB byte is needed.
  if (// The SIB byte must be used if there is an index register or the
      // encoding requires a SIB byte.
      !ForceSIB && IndexReg.getReg() == 0 &&
      // The SIB byte must be used if the base is ESP/RSP/R12, all of which
      // encode to an R/M value of 4, which indicates that a SIB byte is
      // present.
      BaseRegNo != N86::ESP &&
      // If there is no base register and we're in 64-bit mode, we need a SIB
      // byte to emit an addr that is just 'disp32' (the non-RIP relative form).
      (!STI.hasFeature(X86::Is64Bit) || BaseReg != 0)) {

    if (BaseReg == 0) { // [disp32]     in X86-32 mode
      emitByte(modRMByte(0, RegOpcodeField, 5), OS);
      emitImmediate(Disp, MI.getLoc(), 4, FK_Data_4, StartByte, OS, Fixups);
      return;
    }

    // If the base is not EBP/ESP/R12/R13 and there is no displacement, use
    // simple indirect register encoding, this handles addresses like [EAX].
    // The encoding for [EBP] or[R13] with no displacement means [disp32] so we
    // handle it by emitting a displacement of 0 later.
    if (BaseRegNo != N86::EBP) {
      if (Disp.isImm() && Disp.getImm() == 0 && AllowNoDisp) {
        emitByte(modRMByte(0, RegOpcodeField, BaseRegNo), OS);
        return;
      }

      // If the displacement is @tlscall, treat it as a zero.
      if (Disp.isExpr()) {
        auto *Sym = dyn_cast<MCSymbolRefExpr>(Disp.getExpr());
        if (Sym && Sym->getKind() == MCSymbolRefExpr::VK_TLSCALL) {
          // This is exclusively used by call *a@tlscall(base). The relocation
          // (R_386_TLSCALL or R_X86_64_TLSCALL) applies to the beginning.
          Fixups.push_back(MCFixup::create(0, Sym, FK_NONE, MI.getLoc()));
          emitByte(modRMByte(0, RegOpcodeField, BaseRegNo), OS);
          return;
        }
      }
    }

    // Otherwise, if the displacement fits in a byte, encode as [REG+disp8].
    // Including a compressed disp8 for EVEX instructions that support it.
    // This also handles the 0 displacement for [EBP] or [R13]. We can't use
    // disp8 if the {disp32} pseudo prefix is present.
    if (Disp.isImm() && AllowDisp8) {
      int ImmOffset = 0;
      if (isDispOrCDisp8(TSFlags, Disp.getImm(), ImmOffset)) {
        emitByte(modRMByte(1, RegOpcodeField, BaseRegNo), OS);
        emitImmediate(Disp, MI.getLoc(), 1, FK_Data_1, StartByte, OS, Fixups,
                      ImmOffset);
        return;
      }
    }

    // Otherwise, emit the most general non-SIB encoding: [REG+disp32].
    // Displacement may be 0 for [EBP] or [R13] case if {disp32} pseudo prefix
    // prevented using disp8 above.
    emitByte(modRMByte(2, RegOpcodeField, BaseRegNo), OS);
    unsigned Opcode = MI.getOpcode();
    unsigned FixupKind = Opcode == X86::MOV32rm ? X86::reloc_signed_4byte_relax
                                                : X86::reloc_signed_4byte;
    emitImmediate(Disp, MI.getLoc(), 4, MCFixupKind(FixupKind), StartByte, OS,
                  Fixups);
    return;
  }

  // We need a SIB byte, so start by outputting the ModR/M byte first
  assert(IndexReg.getReg() != X86::ESP && IndexReg.getReg() != X86::RSP &&
         "Cannot use ESP as index reg!");

  bool ForceDisp32 = false;
  bool ForceDisp8 = false;
  int ImmOffset = 0;
  if (BaseReg == 0) {
    // If there is no base register, we emit the special case SIB byte with
    // MOD=0, BASE=5, to JUST get the index, scale, and displacement.
    BaseRegNo = 5;
    emitByte(modRMByte(0, RegOpcodeField, 4), OS);
    ForceDisp32 = true;
  } else if (Disp.isImm() && Disp.getImm() == 0 && AllowNoDisp &&
             // Base reg can't be EBP/RBP/R13 as that would end up with '5' as
             // the base field, but that is the magic [*] nomenclature that
             // indicates no base when mod=0. For these cases we'll emit a 0
             // displacement instead.
             BaseRegNo != N86::EBP) {
    // Emit no displacement ModR/M byte
    emitByte(modRMByte(0, RegOpcodeField, 4), OS);
  } else if (Disp.isImm() && AllowDisp8 &&
             isDispOrCDisp8(TSFlags, Disp.getImm(), ImmOffset)) {
    // Displacement fits in a byte or matches an EVEX compressed disp8, use
    // disp8 encoding. This also handles EBP/R13 base with 0 displacement unless
    // {disp32} pseudo prefix was used.
    emitByte(modRMByte(1, RegOpcodeField, 4), OS);
    ForceDisp8 = true;
  } else {
    // Otherwise, emit the normal disp32 encoding.
    emitByte(modRMByte(2, RegOpcodeField, 4), OS);
    ForceDisp32 = true;
  }

  // Calculate what the SS field value should be...
  static const unsigned SSTable[] = {~0U, 0, 1, ~0U, 2, ~0U, ~0U, ~0U, 3};
  unsigned SS = SSTable[Scale.getImm()];

  unsigned IndexRegNo = IndexReg.getReg() ? getX86RegNum(IndexReg) : 4;

  emitSIBByte(SS, IndexRegNo, BaseRegNo, OS);

  // Do we need to output a displacement?
  if (ForceDisp8)
    emitImmediate(Disp, MI.getLoc(), 1, FK_Data_1, StartByte, OS, Fixups,
                  ImmOffset);
  else if (ForceDisp32)
    emitImmediate(Disp, MI.getLoc(), 4, MCFixupKind(X86::reloc_signed_4byte),
                  StartByte, OS, Fixups);
}

/// Emit all instruction prefixes.
///
/// \returns true if REX prefix is used, otherwise returns false.
bool X86MCCodeEmitter::emitPrefixImpl(unsigned &CurOp, const MCInst &MI,
                                      const MCSubtargetInfo &STI,
                                      raw_ostream &OS) const {
  uint64_t TSFlags = MCII.get(MI.getOpcode()).TSFlags;
  // Determine where the memory operand starts, if present.
  int MemoryOperand = X86II::getMemoryOperandNo(TSFlags);
  // Emit segment override opcode prefix as needed.
  if (MemoryOperand != -1) {
    MemoryOperand += CurOp;
    emitSegmentOverridePrefix(MemoryOperand + X86::AddrSegmentReg, MI, OS);
  }

  // Emit the repeat opcode prefix as needed.
  unsigned Flags = MI.getFlags();
  if (TSFlags & X86II::REP || Flags & X86::IP_HAS_REPEAT)
    emitByte(0xF3, OS);
  if (Flags & X86::IP_HAS_REPEAT_NE)
    emitByte(0xF2, OS);

  // Emit the address size opcode prefix as needed.
  if (X86_MC::needsAddressSizeOverride(MI, STI, MemoryOperand, TSFlags) ||
      Flags & X86::IP_HAS_AD_SIZE)
    emitByte(0x67, OS);

  uint64_t Form = TSFlags & X86II::FormMask;
  switch (Form) {
  default:
    break;
  case X86II::RawFrmDstSrc: {
    // Emit segment override opcode prefix as needed (not for %ds).
    if (MI.getOperand(2).getReg() != X86::DS)
      emitSegmentOverridePrefix(2, MI, OS);
    CurOp += 3; // Consume operands.
    break;
  }
  case X86II::RawFrmSrc: {
    // Emit segment override opcode prefix as needed (not for %ds).
    if (MI.getOperand(1).getReg() != X86::DS)
      emitSegmentOverridePrefix(1, MI, OS);
    CurOp += 2; // Consume operands.
    break;
  }
  case X86II::RawFrmDst: {
    ++CurOp; // Consume operand.
    break;
  }
  case X86II::RawFrmMemOffs: {
    // Emit segment override opcode prefix as needed.
    emitSegmentOverridePrefix(1, MI, OS);
    break;
  }
  }

  // REX prefix is optional, but if used must be immediately before the opcode
  // Encoding type for this instruction.
  uint64_t Encoding = TSFlags & X86II::EncodingMask;
  bool HasREX = false;
  if (Encoding)
    emitVEXOpcodePrefix(MemoryOperand, MI, OS);
  else
    HasREX = emitOpcodePrefix(MemoryOperand, MI, STI, OS);

  return HasREX;
}

/// AVX instructions are encoded using a opcode prefix called VEX.
void X86MCCodeEmitter::emitVEXOpcodePrefix(int MemOperand, const MCInst &MI,
                                           raw_ostream &OS) const {
  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  uint64_t TSFlags = Desc.TSFlags;

  assert(!(TSFlags & X86II::LOCK) && "Can't have LOCK VEX.");

  uint64_t Encoding = TSFlags & X86II::EncodingMask;
  bool HasEVEX_K = TSFlags & X86II::EVEX_K;
  bool HasVEX_4V = TSFlags & X86II::VEX_4V;
  bool HasEVEX_RC = TSFlags & X86II::EVEX_RC;

  // VEX_R: opcode externsion equivalent to REX.R in
  // 1's complement (inverted) form
  //
  //  1: Same as REX_R=0 (must be 1 in 32-bit mode)
  //  0: Same as REX_R=1 (64 bit mode only)
  //
  uint8_t VEX_R = 0x1;
  uint8_t EVEX_R2 = 0x1;

  // VEX_X: equivalent to REX.X, only used when a
  // register is used for index in SIB Byte.
  //
  //  1: Same as REX.X=0 (must be 1 in 32-bit mode)
  //  0: Same as REX.X=1 (64-bit mode only)
  uint8_t VEX_X = 0x1;

  // VEX_B:
  //
  //  1: Same as REX_B=0 (ignored in 32-bit mode)
  //  0: Same as REX_B=1 (64 bit mode only)
  //
  uint8_t VEX_B = 0x1;

  // VEX_W: opcode specific (use like REX.W, or used for
  // opcode extension, or ignored, depending on the opcode byte)
  uint8_t VEX_W = (TSFlags & X86II::VEX_W) ? 1 : 0;

  // VEX_5M (VEX m-mmmmm field):
  //
  //  0b00000: Reserved for future use
  //  0b00001: implied 0F leading opcode
  //  0b00010: implied 0F 38 leading opcode bytes
  //  0b00011: implied 0F 3A leading opcode bytes
  //  0b00100: Reserved for future use
  //  0b00101: VEX MAP5
  //  0b00110: VEX MAP6
  //  0b00111-0b11111: Reserved for future use
  //  0b01000: XOP map select - 08h instructions with imm byte
  //  0b01001: XOP map select - 09h instructions with no imm byte
  //  0b01010: XOP map select - 0Ah instructions with imm dword
  uint8_t VEX_5M;
  switch (TSFlags & X86II::OpMapMask) {
  default:
    llvm_unreachable("Invalid prefix!");
  case X86II::TB:
    VEX_5M = 0x1;
    break; // 0F
  case X86II::T8:
    VEX_5M = 0x2;
    break; // 0F 38
  case X86II::TA:
    VEX_5M = 0x3;
    break; // 0F 3A
  case X86II::XOP8:
    VEX_5M = 0x8;
    break;
  case X86II::XOP9:
    VEX_5M = 0x9;
    break;
  case X86II::XOPA:
    VEX_5M = 0xA;
    break;
  case X86II::T_MAP5:
    VEX_5M = 0x5;
    break;
  case X86II::T_MAP6:
    VEX_5M = 0x6;
    break;
  }

  // VEX_4V (VEX vvvv field): a register specifier
  // (in 1's complement form) or 1111 if unused.
  uint8_t VEX_4V = 0xf;
  uint8_t EVEX_V2 = 0x1;

  // EVEX_L2/VEX_L (Vector Length):
  //
  // L2 L
  //  0 0: scalar or 128-bit vector
  //  0 1: 256-bit vector
  //  1 0: 512-bit vector
  //
  uint8_t VEX_L = (TSFlags & X86II::VEX_L) ? 1 : 0;
  uint8_t EVEX_L2 = (TSFlags & X86II::EVEX_L2) ? 1 : 0;

  // VEX_PP: opcode extension providing equivalent
  // functionality of a SIMD prefix
  //
  //  0b00: None
  //  0b01: 66
  //  0b10: F3
  //  0b11: F2
  //
  uint8_t VEX_PP = 0;
  switch (TSFlags & X86II::OpPrefixMask) {
  case X86II::PD:
    VEX_PP = 0x1;
    break; // 66
  case X86II::XS:
    VEX_PP = 0x2;
    break; // F3
  case X86II::XD:
    VEX_PP = 0x3;
    break; // F2
  }

  // EVEX_U
  uint8_t EVEX_U = 1; // Always '1' so far

  // EVEX_z
  uint8_t EVEX_z = (HasEVEX_K && (TSFlags & X86II::EVEX_Z)) ? 1 : 0;

  // EVEX_b
  uint8_t EVEX_b = (TSFlags & X86II::EVEX_B) ? 1 : 0;

  // EVEX_rc
  uint8_t EVEX_rc = 0;

  // EVEX_aaa
  uint8_t EVEX_aaa = 0;

  bool EncodeRC = false;

  // Classify VEX_B, VEX_4V, VEX_R, VEX_X
  unsigned NumOps = Desc.getNumOperands();
  unsigned CurOp = X86II::getOperandBias(Desc);

  switch (TSFlags & X86II::FormMask) {
  default:
    llvm_unreachable("Unexpected form in emitVEXOpcodePrefix!");
  case X86II::MRM_C0:
  case X86II::RawFrm:
  case X86II::PrefixByte:
    break;
  case X86II::MRMDestMemFSIB:
  case X86II::MRMDestMem: {
    // MRMDestMem instructions forms:
    //  MemAddr, src1(ModR/M)
    //  MemAddr, src1(VEX_4V), src2(ModR/M)
    //  MemAddr, src1(ModR/M), imm8
    //
    unsigned BaseRegEnc = getX86RegEncoding(MI, MemOperand + X86::AddrBaseReg);
    VEX_B = ~(BaseRegEnc >> 3) & 1;
    unsigned IndexRegEnc =
        getX86RegEncoding(MI, MemOperand + X86::AddrIndexReg);
    VEX_X = ~(IndexRegEnc >> 3) & 1;
    if (!HasVEX_4V) // Only needed with VSIB which don't use VVVV.
      EVEX_V2 = ~(IndexRegEnc >> 4) & 1;

    CurOp += X86::AddrNumOperands;

    if (HasEVEX_K)
      EVEX_aaa = getX86RegEncoding(MI, CurOp++);

    if (HasVEX_4V) {
      unsigned VRegEnc = getX86RegEncoding(MI, CurOp++);
      VEX_4V = ~VRegEnc & 0xf;
      EVEX_V2 = ~(VRegEnc >> 4) & 1;
    }

    unsigned RegEnc = getX86RegEncoding(MI, CurOp++);
    VEX_R = ~(RegEnc >> 3) & 1;
    EVEX_R2 = ~(RegEnc >> 4) & 1;
    break;
  }
  case X86II::MRMSrcMemFSIB:
  case X86II::MRMSrcMem: {
    // MRMSrcMem instructions forms:
    //  src1(ModR/M), MemAddr
    //  src1(ModR/M), src2(VEX_4V), MemAddr
    //  src1(ModR/M), MemAddr, imm8
    //  src1(ModR/M), MemAddr, src2(Imm[7:4])
    //
    //  FMA4:
    //  dst(ModR/M.reg), src1(VEX_4V), src2(ModR/M), src3(Imm[7:4])
    unsigned RegEnc = getX86RegEncoding(MI, CurOp++);
    VEX_R = ~(RegEnc >> 3) & 1;
    EVEX_R2 = ~(RegEnc >> 4) & 1;

    if (HasEVEX_K)
      EVEX_aaa = getX86RegEncoding(MI, CurOp++);

    if (HasVEX_4V) {
      unsigned VRegEnc = getX86RegEncoding(MI, CurOp++);
      VEX_4V = ~VRegEnc & 0xf;
      EVEX_V2 = ~(VRegEnc >> 4) & 1;
    }

    unsigned BaseRegEnc = getX86RegEncoding(MI, MemOperand + X86::AddrBaseReg);
    VEX_B = ~(BaseRegEnc >> 3) & 1;
    unsigned IndexRegEnc =
        getX86RegEncoding(MI, MemOperand + X86::AddrIndexReg);
    VEX_X = ~(IndexRegEnc >> 3) & 1;
    if (!HasVEX_4V) // Only needed with VSIB which don't use VVVV.
      EVEX_V2 = ~(IndexRegEnc >> 4) & 1;

    break;
  }
  case X86II::MRMSrcMem4VOp3: {
    // Instruction format for 4VOp3:
    //   src1(ModR/M), MemAddr, src3(VEX_4V)
    unsigned RegEnc = getX86RegEncoding(MI, CurOp++);
    VEX_R = ~(RegEnc >> 3) & 1;

    unsigned BaseRegEnc = getX86RegEncoding(MI, MemOperand + X86::AddrBaseReg);
    VEX_B = ~(BaseRegEnc >> 3) & 1;
    unsigned IndexRegEnc =
        getX86RegEncoding(MI, MemOperand + X86::AddrIndexReg);
    VEX_X = ~(IndexRegEnc >> 3) & 1;

    VEX_4V = ~getX86RegEncoding(MI, CurOp + X86::AddrNumOperands) & 0xf;
    break;
  }
  case X86II::MRMSrcMemOp4: {
    //  dst(ModR/M.reg), src1(VEX_4V), src2(Imm[7:4]), src3(ModR/M),
    unsigned RegEnc = getX86RegEncoding(MI, CurOp++);
    VEX_R = ~(RegEnc >> 3) & 1;

    unsigned VRegEnc = getX86RegEncoding(MI, CurOp++);
    VEX_4V = ~VRegEnc & 0xf;

    unsigned BaseRegEnc = getX86RegEncoding(MI, MemOperand + X86::AddrBaseReg);
    VEX_B = ~(BaseRegEnc >> 3) & 1;
    unsigned IndexRegEnc =
        getX86RegEncoding(MI, MemOperand + X86::AddrIndexReg);
    VEX_X = ~(IndexRegEnc >> 3) & 1;
    break;
  }
  case X86II::MRM0m:
  case X86II::MRM1m:
  case X86II::MRM2m:
  case X86II::MRM3m:
  case X86II::MRM4m:
  case X86II::MRM5m:
  case X86II::MRM6m:
  case X86II::MRM7m: {
    // MRM[0-9]m instructions forms:
    //  MemAddr
    //  src1(VEX_4V), MemAddr
    if (HasVEX_4V) {
      unsigned VRegEnc = getX86RegEncoding(MI, CurOp++);
      VEX_4V = ~VRegEnc & 0xf;
      EVEX_V2 = ~(VRegEnc >> 4) & 1;
    }

    if (HasEVEX_K)
      EVEX_aaa = getX86RegEncoding(MI, CurOp++);

    unsigned BaseRegEnc = getX86RegEncoding(MI, MemOperand + X86::AddrBaseReg);
    VEX_B = ~(BaseRegEnc >> 3) & 1;
    unsigned IndexRegEnc =
        getX86RegEncoding(MI, MemOperand + X86::AddrIndexReg);
    VEX_X = ~(IndexRegEnc >> 3) & 1;
    if (!HasVEX_4V) // Only needed with VSIB which don't use VVVV.
      EVEX_V2 = ~(IndexRegEnc >> 4) & 1;

    break;
  }
  case X86II::MRMSrcReg: {
    // MRMSrcReg instructions forms:
    //  dst(ModR/M), src1(VEX_4V), src2(ModR/M), src3(Imm[7:4])
    //  dst(ModR/M), src1(ModR/M)
    //  dst(ModR/M), src1(ModR/M), imm8
    //
    //  FMA4:
    //  dst(ModR/M.reg), src1(VEX_4V), src2(Imm[7:4]), src3(ModR/M),
    unsigned RegEnc = getX86RegEncoding(MI, CurOp++);
    VEX_R = ~(RegEnc >> 3) & 1;
    EVEX_R2 = ~(RegEnc >> 4) & 1;

    if (HasEVEX_K)
      EVEX_aaa = getX86RegEncoding(MI, CurOp++);

    if (HasVEX_4V) {
      unsigned VRegEnc = getX86RegEncoding(MI, CurOp++);
      VEX_4V = ~VRegEnc & 0xf;
      EVEX_V2 = ~(VRegEnc >> 4) & 1;
    }

    RegEnc = getX86RegEncoding(MI, CurOp++);
    VEX_B = ~(RegEnc >> 3) & 1;
    VEX_X = ~(RegEnc >> 4) & 1;

    if (EVEX_b) {
      if (HasEVEX_RC) {
        unsigned RcOperand = NumOps - 1;
        assert(RcOperand >= CurOp);
        EVEX_rc = MI.getOperand(RcOperand).getImm();
        assert(EVEX_rc <= 3 && "Invalid rounding control!");
      }
      EncodeRC = true;
    }
    break;
  }
  case X86II::MRMSrcReg4VOp3: {
    // Instruction format for 4VOp3:
    //   src1(ModR/M), src2(ModR/M), src3(VEX_4V)
    unsigned RegEnc = getX86RegEncoding(MI, CurOp++);
    VEX_R = ~(RegEnc >> 3) & 1;

    RegEnc = getX86RegEncoding(MI, CurOp++);
    VEX_B = ~(RegEnc >> 3) & 1;

    VEX_4V = ~getX86RegEncoding(MI, CurOp++) & 0xf;
    break;
  }
  case X86II::MRMSrcRegOp4: {
    //  dst(ModR/M.reg), src1(VEX_4V), src2(Imm[7:4]), src3(ModR/M),
    unsigned RegEnc = getX86RegEncoding(MI, CurOp++);
    VEX_R = ~(RegEnc >> 3) & 1;

    unsigned VRegEnc = getX86RegEncoding(MI, CurOp++);
    VEX_4V = ~VRegEnc & 0xf;

    // Skip second register source (encoded in Imm[7:4])
    ++CurOp;

    RegEnc = getX86RegEncoding(MI, CurOp++);
    VEX_B = ~(RegEnc >> 3) & 1;
    VEX_X = ~(RegEnc >> 4) & 1;
    break;
  }
  case X86II::MRMDestReg: {
    // MRMDestReg instructions forms:
    //  dst(ModR/M), src(ModR/M)
    //  dst(ModR/M), src(ModR/M), imm8
    //  dst(ModR/M), src1(VEX_4V), src2(ModR/M)
    unsigned RegEnc = getX86RegEncoding(MI, CurOp++);
    VEX_B = ~(RegEnc >> 3) & 1;
    VEX_X = ~(RegEnc >> 4) & 1;

    if (HasEVEX_K)
      EVEX_aaa = getX86RegEncoding(MI, CurOp++);

    if (HasVEX_4V) {
      unsigned VRegEnc = getX86RegEncoding(MI, CurOp++);
      VEX_4V = ~VRegEnc & 0xf;
      EVEX_V2 = ~(VRegEnc >> 4) & 1;
    }

    RegEnc = getX86RegEncoding(MI, CurOp++);
    VEX_R = ~(RegEnc >> 3) & 1;
    EVEX_R2 = ~(RegEnc >> 4) & 1;
    if (EVEX_b)
      EncodeRC = true;
    break;
  }
  case X86II::MRMr0: {
    // MRMr0 instructions forms:
    //  11:rrr:000
    //  dst(ModR/M)
    unsigned RegEnc = getX86RegEncoding(MI, CurOp++);
    VEX_R = ~(RegEnc >> 3) & 1;
    EVEX_R2 = ~(RegEnc >> 4) & 1;
    break;
  }
  case X86II::MRM0r:
  case X86II::MRM1r:
  case X86II::MRM2r:
  case X86II::MRM3r:
  case X86II::MRM4r:
  case X86II::MRM5r:
  case X86II::MRM6r:
  case X86II::MRM7r: {
    // MRM0r-MRM7r instructions forms:
    //  dst(VEX_4V), src(ModR/M), imm8
    if (HasVEX_4V) {
      unsigned VRegEnc = getX86RegEncoding(MI, CurOp++);
      VEX_4V = ~VRegEnc & 0xf;
      EVEX_V2 = ~(VRegEnc >> 4) & 1;
    }
    if (HasEVEX_K)
      EVEX_aaa = getX86RegEncoding(MI, CurOp++);

    unsigned RegEnc = getX86RegEncoding(MI, CurOp++);
    VEX_B = ~(RegEnc >> 3) & 1;
    VEX_X = ~(RegEnc >> 4) & 1;
    break;
  }
  }

  if (Encoding == X86II::VEX || Encoding == X86II::XOP) {
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
    //  XOP uses a similar prefix:
    //    +-----+ +--------------+ +-------------------+
    //    | 8Fh | | RXB | m-mmmm | | W | vvvv | L | pp |
    //    +-----+ +--------------+ +-------------------+
    uint8_t LastByte = VEX_PP | (VEX_L << 2) | (VEX_4V << 3);

    // Can we use the 2 byte VEX prefix?
    if (!(MI.getFlags() & X86::IP_USE_VEX3) && Encoding == X86II::VEX &&
        VEX_B && VEX_X && !VEX_W && (VEX_5M == 1)) {
      emitByte(0xC5, OS);
      emitByte(LastByte | (VEX_R << 7), OS);
      return;
    }

    // 3 byte VEX prefix
    emitByte(Encoding == X86II::XOP ? 0x8F : 0xC4, OS);
    emitByte(VEX_R << 7 | VEX_X << 6 | VEX_B << 5 | VEX_5M, OS);
    emitByte(LastByte | (VEX_W << 7), OS);
  } else {
    assert(Encoding == X86II::EVEX && "unknown encoding!");
    // EVEX opcode prefix can have 4 bytes
    //
    // +-----+ +--------------+ +-------------------+ +------------------------+
    // | 62h | | RXBR' | 0mmm | | W | vvvv | U | pp | | z | L'L | b | v' | aaa |
    // +-----+ +--------------+ +-------------------+ +------------------------+
    assert((VEX_5M & 0x7) == VEX_5M &&
           "More than 3 significant bits in VEX.m-mmmm fields for EVEX!");

    emitByte(0x62, OS);
    emitByte((VEX_R << 7) | (VEX_X << 6) | (VEX_B << 5) | (EVEX_R2 << 4) |
                 VEX_5M,
             OS);
    emitByte((VEX_W << 7) | (VEX_4V << 3) | (EVEX_U << 2) | VEX_PP, OS);
    if (EncodeRC)
      emitByte((EVEX_z << 7) | (EVEX_rc << 5) | (EVEX_b << 4) | (EVEX_V2 << 3) |
                   EVEX_aaa,
               OS);
    else
      emitByte((EVEX_z << 7) | (EVEX_L2 << 6) | (VEX_L << 5) | (EVEX_b << 4) |
                   (EVEX_V2 << 3) | EVEX_aaa,
               OS);
  }
}

/// Emit REX prefix which specifies
///   1) 64-bit instructions,
///   2) non-default operand size, and
///   3) use of X86-64 extended registers.
///
/// \returns true if REX prefix is used, otherwise returns false.
bool X86MCCodeEmitter::emitREXPrefix(int MemOperand, const MCInst &MI,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &OS) const {
  uint8_t REX = [&, MemOperand]() {
    uint8_t REX = 0;
    bool UsesHighByteReg = false;

    const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
    uint64_t TSFlags = Desc.TSFlags;

    if (TSFlags & X86II::REX_W)
      REX |= 1 << 3; // set REX.W

    if (MI.getNumOperands() == 0)
      return REX;

    unsigned NumOps = MI.getNumOperands();
    unsigned CurOp = X86II::getOperandBias(Desc);

    // If it accesses SPL, BPL, SIL, or DIL, then it requires a 0x40 REX prefix.
    for (unsigned i = CurOp; i != NumOps; ++i) {
      const MCOperand &MO = MI.getOperand(i);
      if (MO.isReg()) {
        unsigned Reg = MO.getReg();
        if (Reg == X86::AH || Reg == X86::BH || Reg == X86::CH ||
            Reg == X86::DH)
          UsesHighByteReg = true;
        if (X86II::isX86_64NonExtLowByteReg(Reg))
          // FIXME: The caller of determineREXPrefix slaps this prefix onto
          // anything that returns non-zero.
          REX |= 0x40; // REX fixed encoding prefix
      } else if (MO.isExpr() && STI.getTargetTriple().isX32()) {
        // GOTTPOFF and TLSDESC relocations require a REX prefix to allow
        // linker optimizations: even if the instructions we see may not require
        // any prefix, they may be replaced by instructions that do. This is
        // handled as a special case here so that it also works for hand-written
        // assembly without the user needing to write REX, as with GNU as.
        const auto *Ref = dyn_cast<MCSymbolRefExpr>(MO.getExpr());
        if (Ref && (Ref->getKind() == MCSymbolRefExpr::VK_GOTTPOFF ||
                    Ref->getKind() == MCSymbolRefExpr::VK_TLSDESC)) {
          REX |= 0x40; // REX fixed encoding prefix
        }
      }
    }

    switch (TSFlags & X86II::FormMask) {
    case X86II::AddRegFrm:
      REX |= isREXExtendedReg(MI, CurOp++) << 0; // REX.B
      break;
    case X86II::MRMSrcReg:
    case X86II::MRMSrcRegCC:
      REX |= isREXExtendedReg(MI, CurOp++) << 2; // REX.R
      REX |= isREXExtendedReg(MI, CurOp++) << 0; // REX.B
      break;
    case X86II::MRMSrcMem:
    case X86II::MRMSrcMemCC:
      REX |= isREXExtendedReg(MI, CurOp++) << 2;                        // REX.R
      REX |= isREXExtendedReg(MI, MemOperand + X86::AddrBaseReg) << 0;  // REX.B
      REX |= isREXExtendedReg(MI, MemOperand + X86::AddrIndexReg) << 1; // REX.X
      CurOp += X86::AddrNumOperands;
      break;
    case X86II::MRMDestReg:
      REX |= isREXExtendedReg(MI, CurOp++) << 0; // REX.B
      REX |= isREXExtendedReg(MI, CurOp++) << 2; // REX.R
      break;
    case X86II::MRMDestMem:
      REX |= isREXExtendedReg(MI, MemOperand + X86::AddrBaseReg) << 0;  // REX.B
      REX |= isREXExtendedReg(MI, MemOperand + X86::AddrIndexReg) << 1; // REX.X
      CurOp += X86::AddrNumOperands;
      REX |= isREXExtendedReg(MI, CurOp++) << 2; // REX.R
      break;
    case X86II::MRMXmCC:
    case X86II::MRMXm:
    case X86II::MRM0m:
    case X86II::MRM1m:
    case X86II::MRM2m:
    case X86II::MRM3m:
    case X86II::MRM4m:
    case X86II::MRM5m:
    case X86II::MRM6m:
    case X86II::MRM7m:
      REX |= isREXExtendedReg(MI, MemOperand + X86::AddrBaseReg) << 0;  // REX.B
      REX |= isREXExtendedReg(MI, MemOperand + X86::AddrIndexReg) << 1; // REX.X
      break;
    case X86II::MRMXrCC:
    case X86II::MRMXr:
    case X86II::MRM0r:
    case X86II::MRM1r:
    case X86II::MRM2r:
    case X86II::MRM3r:
    case X86II::MRM4r:
    case X86II::MRM5r:
    case X86II::MRM6r:
    case X86II::MRM7r:
      REX |= isREXExtendedReg(MI, CurOp++) << 0; // REX.B
      break;
    case X86II::MRMr0:
      REX |= isREXExtendedReg(MI, CurOp++) << 2; // REX.R
      break;
    case X86II::MRMDestMemFSIB:
      llvm_unreachable("FSIB format never need REX prefix!");
    }
    if (REX && UsesHighByteReg)
      report_fatal_error(
          "Cannot encode high byte register in REX-prefixed instruction");
    return REX;
  }();

  if (!REX)
    return false;

  emitByte(0x40 | REX, OS);
  return true;
}

/// Emit segment override opcode prefix as needed.
void X86MCCodeEmitter::emitSegmentOverridePrefix(unsigned SegOperand,
                                                 const MCInst &MI,
                                                 raw_ostream &OS) const {
  // Check for explicit segment override on memory operand.
  if (unsigned Reg = MI.getOperand(SegOperand).getReg())
    emitByte(X86::getSegmentOverridePrefixForReg(Reg), OS);
}

/// Emit all instruction prefixes prior to the opcode.
///
/// \param MemOperand the operand # of the start of a memory operand if present.
/// If not present, it is -1.
///
/// \returns true if REX prefix is used, otherwise returns false.
bool X86MCCodeEmitter::emitOpcodePrefix(int MemOperand, const MCInst &MI,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &OS) const {
  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  uint64_t TSFlags = Desc.TSFlags;

  // Emit the operand size opcode prefix as needed.
  if ((TSFlags & X86II::OpSizeMask) ==
      (STI.hasFeature(X86::Is16Bit) ? X86II::OpSize32 : X86II::OpSize16))
    emitByte(0x66, OS);

  // Emit the LOCK opcode prefix.
  if (TSFlags & X86II::LOCK || MI.getFlags() & X86::IP_HAS_LOCK)
    emitByte(0xF0, OS);

  // Emit the NOTRACK opcode prefix.
  if (TSFlags & X86II::NOTRACK || MI.getFlags() & X86::IP_HAS_NOTRACK)
    emitByte(0x3E, OS);

  switch (TSFlags & X86II::OpPrefixMask) {
  case X86II::PD: // 66
    emitByte(0x66, OS);
    break;
  case X86II::XS: // F3
    emitByte(0xF3, OS);
    break;
  case X86II::XD: // F2
    emitByte(0xF2, OS);
    break;
  }

  // Handle REX prefix.
  assert((STI.hasFeature(X86::Is64Bit) || !(TSFlags & X86II::REX_W)) &&
         "REX.W requires 64bit mode.");
  bool HasREX = STI.hasFeature(X86::Is64Bit)
                    ? emitREXPrefix(MemOperand, MI, STI, OS)
                    : false;

  // 0x0F escape code must be emitted just before the opcode.
  switch (TSFlags & X86II::OpMapMask) {
  case X86II::TB:        // Two-byte opcode map
  case X86II::T8:        // 0F 38
  case X86II::TA:        // 0F 3A
  case X86II::ThreeDNow: // 0F 0F, second 0F emitted by caller.
    emitByte(0x0F, OS);
    break;
  }

  switch (TSFlags & X86II::OpMapMask) {
  case X86II::T8: // 0F 38
    emitByte(0x38, OS);
    break;
  case X86II::TA: // 0F 3A
    emitByte(0x3A, OS);
    break;
  }

  return HasREX;
}

void X86MCCodeEmitter::emitPrefix(const MCInst &MI, raw_ostream &OS,
                                  const MCSubtargetInfo &STI) const {
  unsigned Opcode = MI.getOpcode();
  const MCInstrDesc &Desc = MCII.get(Opcode);
  uint64_t TSFlags = Desc.TSFlags;

  // Pseudo instructions don't get encoded.
  if (X86II::isPseudo(TSFlags))
    return;

  unsigned CurOp = X86II::getOperandBias(Desc);

  emitPrefixImpl(CurOp, MI, STI, OS);
}

void X86MCCodeEmitter::encodeInstruction(const MCInst &MI, raw_ostream &OS,
                                         SmallVectorImpl<MCFixup> &Fixups,
                                         const MCSubtargetInfo &STI) const {
  unsigned Opcode = MI.getOpcode();
  const MCInstrDesc &Desc = MCII.get(Opcode);
  uint64_t TSFlags = Desc.TSFlags;

  // Pseudo instructions don't get encoded.
  if (X86II::isPseudo(TSFlags))
    return;

  unsigned NumOps = Desc.getNumOperands();
  unsigned CurOp = X86II::getOperandBias(Desc);

  uint64_t StartByte = OS.tell();

  bool HasREX = emitPrefixImpl(CurOp, MI, STI, OS);

  // It uses the VEX.VVVV field?
  bool HasVEX_4V = TSFlags & X86II::VEX_4V;
  bool HasVEX_I8Reg = (TSFlags & X86II::ImmMask) == X86II::Imm8Reg;

  // It uses the EVEX.aaa field?
  bool HasEVEX_K = TSFlags & X86II::EVEX_K;
  bool HasEVEX_RC = TSFlags & X86II::EVEX_RC;

  // Used if a register is encoded in 7:4 of immediate.
  unsigned I8RegNum = 0;

  uint8_t BaseOpcode = X86II::getBaseOpcodeFor(TSFlags);

  if ((TSFlags & X86II::OpMapMask) == X86II::ThreeDNow)
    BaseOpcode = 0x0F; // Weird 3DNow! encoding.

  unsigned OpcodeOffset = 0;

  uint64_t Form = TSFlags & X86II::FormMask;
  switch (Form) {
  default:
    errs() << "FORM: " << Form << "\n";
    llvm_unreachable("Unknown FormMask value in X86MCCodeEmitter!");
  case X86II::Pseudo:
    llvm_unreachable("Pseudo instruction shouldn't be emitted");
  case X86II::RawFrmDstSrc:
  case X86II::RawFrmSrc:
  case X86II::RawFrmDst:
  case X86II::PrefixByte:
    emitByte(BaseOpcode, OS);
    break;
  case X86II::AddCCFrm: {
    // This will be added to the opcode in the fallthrough.
    OpcodeOffset = MI.getOperand(NumOps - 1).getImm();
    assert(OpcodeOffset < 16 && "Unexpected opcode offset!");
    --NumOps; // Drop the operand from the end.
    LLVM_FALLTHROUGH;
  case X86II::RawFrm:
    emitByte(BaseOpcode + OpcodeOffset, OS);

    if (!STI.hasFeature(X86::Is64Bit) || !isPCRel32Branch(MI, MCII))
      break;

    const MCOperand &Op = MI.getOperand(CurOp++);
    emitImmediate(Op, MI.getLoc(), X86II::getSizeOfImm(TSFlags),
                  MCFixupKind(X86::reloc_branch_4byte_pcrel), StartByte, OS,
                  Fixups);
    break;
  }
  case X86II::RawFrmMemOffs:
    emitByte(BaseOpcode, OS);
    emitImmediate(MI.getOperand(CurOp++), MI.getLoc(),
                  X86II::getSizeOfImm(TSFlags), getImmFixupKind(TSFlags),
                  StartByte, OS, Fixups);
    ++CurOp; // skip segment operand
    break;
  case X86II::RawFrmImm8:
    emitByte(BaseOpcode, OS);
    emitImmediate(MI.getOperand(CurOp++), MI.getLoc(),
                  X86II::getSizeOfImm(TSFlags), getImmFixupKind(TSFlags),
                  StartByte, OS, Fixups);
    emitImmediate(MI.getOperand(CurOp++), MI.getLoc(), 1, FK_Data_1, StartByte,
                  OS, Fixups);
    break;
  case X86II::RawFrmImm16:
    emitByte(BaseOpcode, OS);
    emitImmediate(MI.getOperand(CurOp++), MI.getLoc(),
                  X86II::getSizeOfImm(TSFlags), getImmFixupKind(TSFlags),
                  StartByte, OS, Fixups);
    emitImmediate(MI.getOperand(CurOp++), MI.getLoc(), 2, FK_Data_2, StartByte,
                  OS, Fixups);
    break;

  case X86II::AddRegFrm:
    emitByte(BaseOpcode + getX86RegNum(MI.getOperand(CurOp++)), OS);
    break;

  case X86II::MRMDestReg: {
    emitByte(BaseOpcode, OS);
    unsigned SrcRegNum = CurOp + 1;

    if (HasEVEX_K) // Skip writemask
      ++SrcRegNum;

    if (HasVEX_4V) // Skip 1st src (which is encoded in VEX_VVVV)
      ++SrcRegNum;

    emitRegModRMByte(MI.getOperand(CurOp),
                     getX86RegNum(MI.getOperand(SrcRegNum)), OS);
    CurOp = SrcRegNum + 1;
    break;
  }
  case X86II::MRMDestMemFSIB:
  case X86II::MRMDestMem: {
    emitByte(BaseOpcode, OS);
    unsigned SrcRegNum = CurOp + X86::AddrNumOperands;

    if (HasEVEX_K) // Skip writemask
      ++SrcRegNum;

    if (HasVEX_4V) // Skip 1st src (which is encoded in VEX_VVVV)
      ++SrcRegNum;

    bool ForceSIB = (Form == X86II::MRMDestMemFSIB);
    emitMemModRMByte(MI, CurOp, getX86RegNum(MI.getOperand(SrcRegNum)), TSFlags,
                     HasREX, StartByte, OS, Fixups, STI, ForceSIB);
    CurOp = SrcRegNum + 1;
    break;
  }
  case X86II::MRMSrcReg: {
    emitByte(BaseOpcode, OS);
    unsigned SrcRegNum = CurOp + 1;

    if (HasEVEX_K) // Skip writemask
      ++SrcRegNum;

    if (HasVEX_4V) // Skip 1st src (which is encoded in VEX_VVVV)
      ++SrcRegNum;

    emitRegModRMByte(MI.getOperand(SrcRegNum),
                     getX86RegNum(MI.getOperand(CurOp)), OS);
    CurOp = SrcRegNum + 1;
    if (HasVEX_I8Reg)
      I8RegNum = getX86RegEncoding(MI, CurOp++);
    // do not count the rounding control operand
    if (HasEVEX_RC)
      --NumOps;
    break;
  }
  case X86II::MRMSrcReg4VOp3: {
    emitByte(BaseOpcode, OS);
    unsigned SrcRegNum = CurOp + 1;

    emitRegModRMByte(MI.getOperand(SrcRegNum),
                     getX86RegNum(MI.getOperand(CurOp)), OS);
    CurOp = SrcRegNum + 1;
    ++CurOp; // Encoded in VEX.VVVV
    break;
  }
  case X86II::MRMSrcRegOp4: {
    emitByte(BaseOpcode, OS);
    unsigned SrcRegNum = CurOp + 1;

    // Skip 1st src (which is encoded in VEX_VVVV)
    ++SrcRegNum;

    // Capture 2nd src (which is encoded in Imm[7:4])
    assert(HasVEX_I8Reg && "MRMSrcRegOp4 should imply VEX_I8Reg");
    I8RegNum = getX86RegEncoding(MI, SrcRegNum++);

    emitRegModRMByte(MI.getOperand(SrcRegNum),
                     getX86RegNum(MI.getOperand(CurOp)), OS);
    CurOp = SrcRegNum + 1;
    break;
  }
  case X86II::MRMSrcRegCC: {
    unsigned FirstOp = CurOp++;
    unsigned SecondOp = CurOp++;

    unsigned CC = MI.getOperand(CurOp++).getImm();
    emitByte(BaseOpcode + CC, OS);

    emitRegModRMByte(MI.getOperand(SecondOp),
                     getX86RegNum(MI.getOperand(FirstOp)), OS);
    break;
  }
  case X86II::MRMSrcMemFSIB:
  case X86II::MRMSrcMem: {
    unsigned FirstMemOp = CurOp + 1;

    if (HasEVEX_K) // Skip writemask
      ++FirstMemOp;

    if (HasVEX_4V)
      ++FirstMemOp; // Skip the register source (which is encoded in VEX_VVVV).

    emitByte(BaseOpcode, OS);

    bool ForceSIB = (Form == X86II::MRMSrcMemFSIB);
    emitMemModRMByte(MI, FirstMemOp, getX86RegNum(MI.getOperand(CurOp)),
                     TSFlags, HasREX, StartByte, OS, Fixups, STI, ForceSIB);
    CurOp = FirstMemOp + X86::AddrNumOperands;
    if (HasVEX_I8Reg)
      I8RegNum = getX86RegEncoding(MI, CurOp++);
    break;
  }
  case X86II::MRMSrcMem4VOp3: {
    unsigned FirstMemOp = CurOp + 1;

    emitByte(BaseOpcode, OS);

    emitMemModRMByte(MI, FirstMemOp, getX86RegNum(MI.getOperand(CurOp)),
                     TSFlags, HasREX, StartByte, OS, Fixups, STI);
    CurOp = FirstMemOp + X86::AddrNumOperands;
    ++CurOp; // Encoded in VEX.VVVV.
    break;
  }
  case X86II::MRMSrcMemOp4: {
    unsigned FirstMemOp = CurOp + 1;

    ++FirstMemOp; // Skip the register source (which is encoded in VEX_VVVV).

    // Capture second register source (encoded in Imm[7:4])
    assert(HasVEX_I8Reg && "MRMSrcRegOp4 should imply VEX_I8Reg");
    I8RegNum = getX86RegEncoding(MI, FirstMemOp++);

    emitByte(BaseOpcode, OS);

    emitMemModRMByte(MI, FirstMemOp, getX86RegNum(MI.getOperand(CurOp)),
                     TSFlags, HasREX, StartByte, OS, Fixups, STI);
    CurOp = FirstMemOp + X86::AddrNumOperands;
    break;
  }
  case X86II::MRMSrcMemCC: {
    unsigned RegOp = CurOp++;
    unsigned FirstMemOp = CurOp;
    CurOp = FirstMemOp + X86::AddrNumOperands;

    unsigned CC = MI.getOperand(CurOp++).getImm();
    emitByte(BaseOpcode + CC, OS);

    emitMemModRMByte(MI, FirstMemOp, getX86RegNum(MI.getOperand(RegOp)),
                     TSFlags, HasREX, StartByte, OS, Fixups, STI);
    break;
  }

  case X86II::MRMXrCC: {
    unsigned RegOp = CurOp++;

    unsigned CC = MI.getOperand(CurOp++).getImm();
    emitByte(BaseOpcode + CC, OS);
    emitRegModRMByte(MI.getOperand(RegOp), 0, OS);
    break;
  }

  case X86II::MRMXr:
  case X86II::MRM0r:
  case X86II::MRM1r:
  case X86II::MRM2r:
  case X86II::MRM3r:
  case X86II::MRM4r:
  case X86II::MRM5r:
  case X86II::MRM6r:
  case X86II::MRM7r:
    if (HasVEX_4V) // Skip the register dst (which is encoded in VEX_VVVV).
      ++CurOp;
    if (HasEVEX_K) // Skip writemask
      ++CurOp;
    emitByte(BaseOpcode, OS);
    emitRegModRMByte(MI.getOperand(CurOp++),
                     (Form == X86II::MRMXr) ? 0 : Form - X86II::MRM0r, OS);
    break;
  case X86II::MRMr0:
    emitByte(BaseOpcode, OS);
    emitByte(modRMByte(3, getX86RegNum(MI.getOperand(CurOp++)),0), OS);
    break;

  case X86II::MRMXmCC: {
    unsigned FirstMemOp = CurOp;
    CurOp = FirstMemOp + X86::AddrNumOperands;

    unsigned CC = MI.getOperand(CurOp++).getImm();
    emitByte(BaseOpcode + CC, OS);

    emitMemModRMByte(MI, FirstMemOp, 0, TSFlags, HasREX, StartByte, OS, Fixups,
                     STI);
    break;
  }

  case X86II::MRMXm:
  case X86II::MRM0m:
  case X86II::MRM1m:
  case X86II::MRM2m:
  case X86II::MRM3m:
  case X86II::MRM4m:
  case X86II::MRM5m:
  case X86II::MRM6m:
  case X86II::MRM7m:
    if (HasVEX_4V) // Skip the register dst (which is encoded in VEX_VVVV).
      ++CurOp;
    if (HasEVEX_K) // Skip writemask
      ++CurOp;
    emitByte(BaseOpcode, OS);
    emitMemModRMByte(MI, CurOp,
                     (Form == X86II::MRMXm) ? 0 : Form - X86II::MRM0m, TSFlags,
                     HasREX, StartByte, OS, Fixups, STI);
    CurOp += X86::AddrNumOperands;
    break;

  case X86II::MRM0X:
  case X86II::MRM1X:
  case X86II::MRM2X:
  case X86II::MRM3X:
  case X86II::MRM4X:
  case X86II::MRM5X:
  case X86II::MRM6X:
  case X86II::MRM7X:
    emitByte(BaseOpcode, OS);
    emitByte(0xC0 + ((Form - X86II::MRM0X) << 3), OS);
    break;

  case X86II::MRM_C0:
  case X86II::MRM_C1:
  case X86II::MRM_C2:
  case X86II::MRM_C3:
  case X86II::MRM_C4:
  case X86II::MRM_C5:
  case X86II::MRM_C6:
  case X86II::MRM_C7:
  case X86II::MRM_C8:
  case X86II::MRM_C9:
  case X86II::MRM_CA:
  case X86II::MRM_CB:
  case X86II::MRM_CC:
  case X86II::MRM_CD:
  case X86II::MRM_CE:
  case X86II::MRM_CF:
  case X86II::MRM_D0:
  case X86II::MRM_D1:
  case X86II::MRM_D2:
  case X86II::MRM_D3:
  case X86II::MRM_D4:
  case X86II::MRM_D5:
  case X86II::MRM_D6:
  case X86II::MRM_D7:
  case X86II::MRM_D8:
  case X86II::MRM_D9:
  case X86II::MRM_DA:
  case X86II::MRM_DB:
  case X86II::MRM_DC:
  case X86II::MRM_DD:
  case X86II::MRM_DE:
  case X86II::MRM_DF:
  case X86II::MRM_E0:
  case X86II::MRM_E1:
  case X86II::MRM_E2:
  case X86II::MRM_E3:
  case X86II::MRM_E4:
  case X86II::MRM_E5:
  case X86II::MRM_E6:
  case X86II::MRM_E7:
  case X86II::MRM_E8:
  case X86II::MRM_E9:
  case X86II::MRM_EA:
  case X86II::MRM_EB:
  case X86II::MRM_EC:
  case X86II::MRM_ED:
  case X86II::MRM_EE:
  case X86II::MRM_EF:
  case X86II::MRM_F0:
  case X86II::MRM_F1:
  case X86II::MRM_F2:
  case X86II::MRM_F3:
  case X86II::MRM_F4:
  case X86II::MRM_F5:
  case X86II::MRM_F6:
  case X86II::MRM_F7:
  case X86II::MRM_F8:
  case X86II::MRM_F9:
  case X86II::MRM_FA:
  case X86II::MRM_FB:
  case X86II::MRM_FC:
  case X86II::MRM_FD:
  case X86II::MRM_FE:
  case X86II::MRM_FF:
    emitByte(BaseOpcode, OS);
    emitByte(0xC0 + Form - X86II::MRM_C0, OS);
    break;
  }

  if (HasVEX_I8Reg) {
    // The last source register of a 4 operand instruction in AVX is encoded
    // in bits[7:4] of a immediate byte.
    assert(I8RegNum < 16 && "Register encoding out of range");
    I8RegNum <<= 4;
    if (CurOp != NumOps) {
      unsigned Val = MI.getOperand(CurOp++).getImm();
      assert(Val < 16 && "Immediate operand value out of range");
      I8RegNum |= Val;
    }
    emitImmediate(MCOperand::createImm(I8RegNum), MI.getLoc(), 1, FK_Data_1,
                  StartByte, OS, Fixups);
  } else {
    // If there is a remaining operand, it must be a trailing immediate. Emit it
    // according to the right size for the instruction. Some instructions
    // (SSE4a extrq and insertq) have two trailing immediates.
    while (CurOp != NumOps && NumOps - CurOp <= 2) {
      emitImmediate(MI.getOperand(CurOp++), MI.getLoc(),
                    X86II::getSizeOfImm(TSFlags), getImmFixupKind(TSFlags),
                    StartByte, OS, Fixups);
    }
  }

  if ((TSFlags & X86II::OpMapMask) == X86II::ThreeDNow)
    emitByte(X86II::getBaseOpcodeFor(TSFlags), OS);

  assert(OS.tell() - StartByte <= 15 &&
         "The size of instruction must be no longer than 15.");
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

MCCodeEmitter *llvm::createX86MCCodeEmitter(const MCInstrInfo &MCII,
                                            MCContext &Ctx) {
  return new X86MCCodeEmitter(MCII, Ctx);
}
