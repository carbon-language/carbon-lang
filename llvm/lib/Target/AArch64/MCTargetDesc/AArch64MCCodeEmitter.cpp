//=- AArch64/AArch64MCCodeEmitter.cpp - Convert AArch64 code to machine code =//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64MCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mccodeemitter"
#include "MCTargetDesc/AArch64FixupKinds.h"
#include "MCTargetDesc/AArch64MCExpr.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
class AArch64MCCodeEmitter : public MCCodeEmitter {
  AArch64MCCodeEmitter(const AArch64MCCodeEmitter &) LLVM_DELETED_FUNCTION;
  void operator=(const AArch64MCCodeEmitter &) LLVM_DELETED_FUNCTION;
  MCContext &Ctx;

public:
  AArch64MCCodeEmitter(MCContext &ctx) : Ctx(ctx) {}

  ~AArch64MCCodeEmitter() {}

  unsigned getAddSubImmOpValue(const MCInst &MI, unsigned OpIdx,
                               SmallVectorImpl<MCFixup> &Fixups) const;

  unsigned getAdrpLabelOpValue(const MCInst &MI, unsigned OpIdx,
                               SmallVectorImpl<MCFixup> &Fixups) const;

  template<int MemSize>
  unsigned getOffsetUImm12OpValue(const MCInst &MI, unsigned OpIdx,
                                    SmallVectorImpl<MCFixup> &Fixups) const {
    return getOffsetUImm12OpValue(MI, OpIdx, Fixups, MemSize);
  }

  unsigned getOffsetUImm12OpValue(const MCInst &MI, unsigned OpIdx,
                                    SmallVectorImpl<MCFixup> &Fixups,
                                    int MemSize) const;

  unsigned getBitfield32LSLOpValue(const MCInst &MI, unsigned OpIdx,
                                   SmallVectorImpl<MCFixup> &Fixups) const;
  unsigned getBitfield64LSLOpValue(const MCInst &MI, unsigned OpIdx,
                                   SmallVectorImpl<MCFixup> &Fixups) const;


  // Labels are handled mostly the same way: a symbol is needed, and
  // just gets some fixup attached.
  template<AArch64::Fixups fixupDesired>
  unsigned getLabelOpValue(const MCInst &MI, unsigned OpIdx,
                           SmallVectorImpl<MCFixup> &Fixups) const;

  unsigned  getLoadLitLabelOpValue(const MCInst &MI, unsigned OpIdx,
                                   SmallVectorImpl<MCFixup> &Fixups) const;


  unsigned getMoveWideImmOpValue(const MCInst &MI, unsigned OpIdx,
                                 SmallVectorImpl<MCFixup> &Fixups) const;


  unsigned getAddressWithFixup(const MCOperand &MO,
                               unsigned FixupKind,
                               SmallVectorImpl<MCFixup> &Fixups) const;


  // getBinaryCodeForInstr - TableGen'erated function for getting the
  // binary encoding for an instruction.
  uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups) const;

  /// getMachineOpValue - Return binary encoding of operand. If the machine
  /// operand requires relocation, record the relocation and return zero.
  unsigned getMachineOpValue(const MCInst &MI,const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups) const;


  void EmitByte(unsigned char C, raw_ostream &OS) const {
    OS << (char)C;
  }

  void EmitInstruction(uint32_t Val, raw_ostream &OS) const {
    // Output the constant in little endian byte order.
    for (unsigned i = 0; i != 4; ++i) {
      EmitByte(Val & 0xff, OS);
      Val >>= 8;
    }
  }


  void EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups) const;

  template<int hasRs, int hasRt2> unsigned
  fixLoadStoreExclusive(const MCInst &MI, unsigned EncodedValue) const;

  unsigned fixMOVZ(const MCInst &MI, unsigned EncodedValue) const;

  unsigned fixMulHigh(const MCInst &MI, unsigned EncodedValue) const;


};

} // end anonymous namespace

unsigned AArch64MCCodeEmitter::getAddressWithFixup(const MCOperand &MO,
                                       unsigned FixupKind,
                                       SmallVectorImpl<MCFixup> &Fixups) const {
  if (!MO.isExpr()) {
    // This can occur for manually decoded or constructed MCInsts, but neither
    // the assembly-parser nor instruction selection will currently produce an
    // MCInst that's not a symbol reference.
    assert(MO.isImm() && "Unexpected address requested");
    return MO.getImm();
  }

  const MCExpr *Expr = MO.getExpr();
  MCFixupKind Kind = MCFixupKind(FixupKind);
  Fixups.push_back(MCFixup::Create(0, Expr, Kind));

  return 0;
}

unsigned AArch64MCCodeEmitter::
getOffsetUImm12OpValue(const MCInst &MI, unsigned OpIdx,
                       SmallVectorImpl<MCFixup> &Fixups,
                       int MemSize) const {
  const MCOperand &ImmOp = MI.getOperand(OpIdx);
  if (ImmOp.isImm())
    return ImmOp.getImm();

  assert(ImmOp.isExpr() && "Unexpected operand type");
  const AArch64MCExpr *Expr = cast<AArch64MCExpr>(ImmOp.getExpr());
  unsigned FixupKind;


  switch (Expr->getKind()) {
  default: llvm_unreachable("Unexpected operand modifier");
  case AArch64MCExpr::VK_AARCH64_LO12: {
    unsigned FixupsBySize[] = { AArch64::fixup_a64_ldst8_lo12,
                                AArch64::fixup_a64_ldst16_lo12,
                                AArch64::fixup_a64_ldst32_lo12,
                                AArch64::fixup_a64_ldst64_lo12,
                                AArch64::fixup_a64_ldst128_lo12 };
    assert(MemSize <= 16 && "Invalid fixup for operation");
    FixupKind = FixupsBySize[Log2_32(MemSize)];
    break;
  }
  case AArch64MCExpr::VK_AARCH64_GOT_LO12:
    assert(MemSize == 8 && "Invalid fixup for operation");
    FixupKind = AArch64::fixup_a64_ld64_got_lo12_nc;
    break;
  case AArch64MCExpr::VK_AARCH64_DTPREL_LO12:  {
    unsigned FixupsBySize[] = { AArch64::fixup_a64_ldst8_dtprel_lo12,
                                AArch64::fixup_a64_ldst16_dtprel_lo12,
                                AArch64::fixup_a64_ldst32_dtprel_lo12,
                                AArch64::fixup_a64_ldst64_dtprel_lo12 };
    assert(MemSize <= 8 && "Invalid fixup for operation");
    FixupKind = FixupsBySize[Log2_32(MemSize)];
    break;
  }
  case AArch64MCExpr::VK_AARCH64_DTPREL_LO12_NC: {
    unsigned FixupsBySize[] = { AArch64::fixup_a64_ldst8_dtprel_lo12_nc,
                                AArch64::fixup_a64_ldst16_dtprel_lo12_nc,
                                AArch64::fixup_a64_ldst32_dtprel_lo12_nc,
                                AArch64::fixup_a64_ldst64_dtprel_lo12_nc };
    assert(MemSize <= 8 && "Invalid fixup for operation");
    FixupKind = FixupsBySize[Log2_32(MemSize)];
    break;
  }
  case AArch64MCExpr::VK_AARCH64_GOTTPREL_LO12:
    assert(MemSize == 8 && "Invalid fixup for operation");
    FixupKind = AArch64::fixup_a64_ld64_gottprel_lo12_nc;
    break;
  case AArch64MCExpr::VK_AARCH64_TPREL_LO12:{
    unsigned FixupsBySize[] = { AArch64::fixup_a64_ldst8_tprel_lo12,
                                AArch64::fixup_a64_ldst16_tprel_lo12,
                                AArch64::fixup_a64_ldst32_tprel_lo12,
                                AArch64::fixup_a64_ldst64_tprel_lo12 };
    assert(MemSize <= 8 && "Invalid fixup for operation");
    FixupKind = FixupsBySize[Log2_32(MemSize)];
    break;
  }
  case AArch64MCExpr::VK_AARCH64_TPREL_LO12_NC: {
    unsigned FixupsBySize[] = { AArch64::fixup_a64_ldst8_tprel_lo12_nc,
                                AArch64::fixup_a64_ldst16_tprel_lo12_nc,
                                AArch64::fixup_a64_ldst32_tprel_lo12_nc,
                                AArch64::fixup_a64_ldst64_tprel_lo12_nc };
    assert(MemSize <= 8 && "Invalid fixup for operation");
    FixupKind = FixupsBySize[Log2_32(MemSize)];
    break;
  }
  case AArch64MCExpr::VK_AARCH64_TLSDESC_LO12:
    assert(MemSize == 8 && "Invalid fixup for operation");
    FixupKind = AArch64::fixup_a64_tlsdesc_ld64_lo12_nc;
    break;
  }

  return getAddressWithFixup(ImmOp, FixupKind, Fixups);
}

unsigned
AArch64MCCodeEmitter::getAddSubImmOpValue(const MCInst &MI, unsigned OpIdx,
                                       SmallVectorImpl<MCFixup> &Fixups) const {
  const MCOperand &MO = MI.getOperand(OpIdx);
  if (MO.isImm())
    return static_cast<unsigned>(MO.getImm());

  assert(MO.isExpr());

  unsigned FixupKind = 0;
  switch(cast<AArch64MCExpr>(MO.getExpr())->getKind()) {
  default: llvm_unreachable("Invalid expression modifier");
  case AArch64MCExpr::VK_AARCH64_LO12:
    FixupKind = AArch64::fixup_a64_add_lo12; break;
  case AArch64MCExpr::VK_AARCH64_DTPREL_HI12:
    FixupKind = AArch64::fixup_a64_add_dtprel_hi12; break;
  case AArch64MCExpr::VK_AARCH64_DTPREL_LO12:
    FixupKind = AArch64::fixup_a64_add_dtprel_lo12; break;
  case AArch64MCExpr::VK_AARCH64_DTPREL_LO12_NC:
    FixupKind = AArch64::fixup_a64_add_dtprel_lo12_nc; break;
  case AArch64MCExpr::VK_AARCH64_TPREL_HI12:
    FixupKind = AArch64::fixup_a64_add_tprel_hi12; break;
  case AArch64MCExpr::VK_AARCH64_TPREL_LO12:
    FixupKind = AArch64::fixup_a64_add_tprel_lo12; break;
  case AArch64MCExpr::VK_AARCH64_TPREL_LO12_NC:
    FixupKind = AArch64::fixup_a64_add_tprel_lo12_nc; break;
  case AArch64MCExpr::VK_AARCH64_TLSDESC_LO12:
    FixupKind = AArch64::fixup_a64_tlsdesc_add_lo12_nc; break;
  }

  return getAddressWithFixup(MO, FixupKind, Fixups);
}

unsigned
AArch64MCCodeEmitter::getAdrpLabelOpValue(const MCInst &MI, unsigned OpIdx,
                                       SmallVectorImpl<MCFixup> &Fixups) const {

  const MCOperand &MO = MI.getOperand(OpIdx);
  if (MO.isImm())
    return static_cast<unsigned>(MO.getImm());

  assert(MO.isExpr());

  unsigned Modifier = AArch64MCExpr::VK_AARCH64_None;
  if (const AArch64MCExpr *Expr = dyn_cast<AArch64MCExpr>(MO.getExpr()))
    Modifier = Expr->getKind();

  unsigned FixupKind = 0;
  switch(Modifier) {
  case AArch64MCExpr::VK_AARCH64_None:
    FixupKind = AArch64::fixup_a64_adr_prel_page;
    break;
  case AArch64MCExpr::VK_AARCH64_GOT:
    FixupKind = AArch64::fixup_a64_adr_prel_got_page;
    break;
  case AArch64MCExpr::VK_AARCH64_GOTTPREL:
    FixupKind = AArch64::fixup_a64_adr_gottprel_page;
    break;
  case AArch64MCExpr::VK_AARCH64_TLSDESC:
    FixupKind = AArch64::fixup_a64_tlsdesc_adr_page;
    break;
  default:
    llvm_unreachable("Unknown symbol reference kind for ADRP instruction");
  }

  return getAddressWithFixup(MO, FixupKind, Fixups);
}

unsigned
AArch64MCCodeEmitter::getBitfield32LSLOpValue(const MCInst &MI, unsigned OpIdx,
                                       SmallVectorImpl<MCFixup> &Fixups) const {

  const MCOperand &MO = MI.getOperand(OpIdx);
  assert(MO.isImm() && "Only immediate expected for shift");

  return ((32 - MO.getImm()) & 0x1f) | (31 - MO.getImm()) << 6;
}

unsigned
AArch64MCCodeEmitter::getBitfield64LSLOpValue(const MCInst &MI, unsigned OpIdx,
                                       SmallVectorImpl<MCFixup> &Fixups) const {

  const MCOperand &MO = MI.getOperand(OpIdx);
  assert(MO.isImm() && "Only immediate expected for shift");

  return ((64 - MO.getImm()) & 0x3f) | (63 - MO.getImm()) << 6;
}


template<AArch64::Fixups fixupDesired> unsigned
AArch64MCCodeEmitter::getLabelOpValue(const MCInst &MI,
                                      unsigned OpIdx,
                                      SmallVectorImpl<MCFixup> &Fixups) const {
  const MCOperand &MO = MI.getOperand(OpIdx);

  if (MO.isExpr())
    return getAddressWithFixup(MO, fixupDesired, Fixups);

  assert(MO.isImm());
  return MO.getImm();
}

unsigned
AArch64MCCodeEmitter::getLoadLitLabelOpValue(const MCInst &MI,
                                       unsigned OpIdx,
                                       SmallVectorImpl<MCFixup> &Fixups) const {
  const MCOperand &MO = MI.getOperand(OpIdx);

  if (MO.isImm())
    return MO.getImm();

  assert(MO.isExpr());

  unsigned FixupKind;
  if (isa<AArch64MCExpr>(MO.getExpr())) {
    assert(dyn_cast<AArch64MCExpr>(MO.getExpr())->getKind()
           == AArch64MCExpr::VK_AARCH64_GOTTPREL
           && "Invalid symbol modifier for literal load");
    FixupKind = AArch64::fixup_a64_ld_gottprel_prel19;
  } else {
    FixupKind = AArch64::fixup_a64_ld_prel;
  }

  return getAddressWithFixup(MO, FixupKind, Fixups);
}


unsigned
AArch64MCCodeEmitter::getMachineOpValue(const MCInst &MI,
                                       const MCOperand &MO,
                                       SmallVectorImpl<MCFixup> &Fixups) const {
  if (MO.isReg()) {
    return Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());
  } else if (MO.isImm()) {
    return static_cast<unsigned>(MO.getImm());
  }

  llvm_unreachable("Unable to encode MCOperand!");
  return 0;
}

unsigned
AArch64MCCodeEmitter::getMoveWideImmOpValue(const MCInst &MI, unsigned OpIdx,
                                       SmallVectorImpl<MCFixup> &Fixups) const {
  const MCOperand &UImm16MO = MI.getOperand(OpIdx);
  const MCOperand &ShiftMO = MI.getOperand(OpIdx + 1);

  unsigned Result = static_cast<unsigned>(ShiftMO.getImm()) << 16;

  if (UImm16MO.isImm()) {
    Result |= UImm16MO.getImm();
    return Result;
  }

  const AArch64MCExpr *A64E = cast<AArch64MCExpr>(UImm16MO.getExpr());
  AArch64::Fixups requestedFixup;
  switch (A64E->getKind()) {
  default: llvm_unreachable("unexpected expression modifier");
  case AArch64MCExpr::VK_AARCH64_ABS_G0:
    requestedFixup = AArch64::fixup_a64_movw_uabs_g0; break;
  case AArch64MCExpr::VK_AARCH64_ABS_G0_NC:
    requestedFixup = AArch64::fixup_a64_movw_uabs_g0_nc; break;
  case AArch64MCExpr::VK_AARCH64_ABS_G1:
    requestedFixup = AArch64::fixup_a64_movw_uabs_g1; break;
  case AArch64MCExpr::VK_AARCH64_ABS_G1_NC:
    requestedFixup = AArch64::fixup_a64_movw_uabs_g1_nc; break;
  case AArch64MCExpr::VK_AARCH64_ABS_G2:
    requestedFixup = AArch64::fixup_a64_movw_uabs_g2; break;
  case AArch64MCExpr::VK_AARCH64_ABS_G2_NC:
    requestedFixup = AArch64::fixup_a64_movw_uabs_g2_nc; break;
  case AArch64MCExpr::VK_AARCH64_ABS_G3:
    requestedFixup = AArch64::fixup_a64_movw_uabs_g3; break;
  case AArch64MCExpr::VK_AARCH64_SABS_G0:
    requestedFixup = AArch64::fixup_a64_movw_sabs_g0; break;
  case AArch64MCExpr::VK_AARCH64_SABS_G1:
    requestedFixup = AArch64::fixup_a64_movw_sabs_g1; break;
  case AArch64MCExpr::VK_AARCH64_SABS_G2:
    requestedFixup = AArch64::fixup_a64_movw_sabs_g2; break;
  case AArch64MCExpr::VK_AARCH64_DTPREL_G2:
    requestedFixup = AArch64::fixup_a64_movw_dtprel_g2; break;
  case AArch64MCExpr::VK_AARCH64_DTPREL_G1:
    requestedFixup = AArch64::fixup_a64_movw_dtprel_g1; break;
  case AArch64MCExpr::VK_AARCH64_DTPREL_G1_NC:
    requestedFixup = AArch64::fixup_a64_movw_dtprel_g1_nc; break;
  case AArch64MCExpr::VK_AARCH64_DTPREL_G0:
    requestedFixup = AArch64::fixup_a64_movw_dtprel_g0; break;
  case AArch64MCExpr::VK_AARCH64_DTPREL_G0_NC:
    requestedFixup = AArch64::fixup_a64_movw_dtprel_g0_nc; break;
  case AArch64MCExpr::VK_AARCH64_GOTTPREL_G1:
    requestedFixup = AArch64::fixup_a64_movw_gottprel_g1; break;
  case AArch64MCExpr::VK_AARCH64_GOTTPREL_G0_NC:
    requestedFixup = AArch64::fixup_a64_movw_gottprel_g0_nc; break;
  case AArch64MCExpr::VK_AARCH64_TPREL_G2:
    requestedFixup = AArch64::fixup_a64_movw_tprel_g2; break;
  case AArch64MCExpr::VK_AARCH64_TPREL_G1:
    requestedFixup = AArch64::fixup_a64_movw_tprel_g1; break;
  case AArch64MCExpr::VK_AARCH64_TPREL_G1_NC:
    requestedFixup = AArch64::fixup_a64_movw_tprel_g1_nc; break;
  case AArch64MCExpr::VK_AARCH64_TPREL_G0:
    requestedFixup = AArch64::fixup_a64_movw_tprel_g0; break;
  case AArch64MCExpr::VK_AARCH64_TPREL_G0_NC:
    requestedFixup = AArch64::fixup_a64_movw_tprel_g0_nc; break;
  }

  return Result | getAddressWithFixup(UImm16MO, requestedFixup, Fixups);
}

template<int hasRs, int hasRt2> unsigned
AArch64MCCodeEmitter::fixLoadStoreExclusive(const MCInst &MI,
                                            unsigned EncodedValue) const {
  if (!hasRs) EncodedValue |= 0x001F0000;
  if (!hasRt2) EncodedValue |= 0x00007C00;

  return EncodedValue;
}

unsigned
AArch64MCCodeEmitter::fixMOVZ(const MCInst &MI, unsigned EncodedValue) const {
  // If one of the signed fixup kinds is applied to a MOVZ instruction, the
  // eventual result could be either a MOVZ or a MOVN. It's the MCCodeEmitter's
  // job to ensure that any bits possibly affected by this are 0. This means we
  // must zero out bit 30 (essentially emitting a MOVN).
  MCOperand UImm16MO = MI.getOperand(1);

  // Nothing to do if there's no fixup.
  if (UImm16MO.isImm())
    return EncodedValue;

  const AArch64MCExpr *A64E = cast<AArch64MCExpr>(UImm16MO.getExpr());
  switch (A64E->getKind()) {
  case AArch64MCExpr::VK_AARCH64_SABS_G0:
  case AArch64MCExpr::VK_AARCH64_SABS_G1:
  case AArch64MCExpr::VK_AARCH64_SABS_G2:
  case AArch64MCExpr::VK_AARCH64_DTPREL_G2:
  case AArch64MCExpr::VK_AARCH64_DTPREL_G1:
  case AArch64MCExpr::VK_AARCH64_DTPREL_G0:
  case AArch64MCExpr::VK_AARCH64_GOTTPREL_G1:
  case AArch64MCExpr::VK_AARCH64_TPREL_G2:
  case AArch64MCExpr::VK_AARCH64_TPREL_G1:
  case AArch64MCExpr::VK_AARCH64_TPREL_G0:
    return EncodedValue & ~(1u << 30);
  default:
    // Nothing to do for an unsigned fixup.
    return EncodedValue;
  }

  llvm_unreachable("Should have returned by now");
}

unsigned
AArch64MCCodeEmitter::fixMulHigh(const MCInst &MI,
                                 unsigned EncodedValue) const {
  // The Ra field of SMULH and UMULH is unused: it should be assembled as 31
  // (i.e. all bits 1) but is ignored by the processor.
  EncodedValue |= 0x1f << 10;
  return EncodedValue;
}

MCCodeEmitter *llvm::createAArch64MCCodeEmitter(const MCInstrInfo &MCII,
                                                const MCRegisterInfo &MRI,
                                                const MCSubtargetInfo &STI,
                                                MCContext &Ctx) {
  return new AArch64MCCodeEmitter(Ctx);
}

void AArch64MCCodeEmitter::
EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                  SmallVectorImpl<MCFixup> &Fixups) const {
  if (MI.getOpcode() == AArch64::TLSDESCCALL) {
    // This is a directive which applies an R_AARCH64_TLSDESC_CALL to the
    // following (BLR) instruction. It doesn't emit any code itself so it
    // doesn't go through the normal TableGenerated channels.
    MCFixupKind Fixup = MCFixupKind(AArch64::fixup_a64_tlsdesc_call);
    const MCExpr *Expr;
    Expr = AArch64MCExpr::CreateTLSDesc(MI.getOperand(0).getExpr(), Ctx);
    Fixups.push_back(MCFixup::Create(0, Expr, Fixup));
    return;
  }

  uint32_t Binary = getBinaryCodeForInstr(MI, Fixups);

  EmitInstruction(Binary, OS);
}


#include "AArch64GenMCCodeEmitter.inc"
