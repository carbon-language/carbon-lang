//===-- AArch64AsmBackend.cpp - AArch64 Assembler Backend -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the AArch64 implementation of the MCAsmBackend class,
// which is principally concerned with relaxation of the various fixup kinds.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AArch64FixupKinds.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
class AArch64AsmBackend : public MCAsmBackend {
  const MCSubtargetInfo* STI;
public:
  AArch64AsmBackend(const Target &T, const StringRef TT)
    : MCAsmBackend(),
      STI(AArch64_MC::createAArch64MCSubtargetInfo(TT, "", ""))
    {}


  ~AArch64AsmBackend() {
    delete STI;
  }

  bool writeNopData(uint64_t Count, MCObjectWriter *OW) const;

  virtual void processFixupValue(const MCAssembler &Asm,
                                 const MCAsmLayout &Layout,
                                 const MCFixup &Fixup, const MCFragment *DF,
                                 MCValue &Target, uint64_t &Value,
                                 bool &IsResolved);
};
} // end anonymous namespace

void AArch64AsmBackend::processFixupValue(const MCAssembler &Asm,
                                          const MCAsmLayout &Layout,
                                          const MCFixup &Fixup,
                                          const MCFragment *DF,
                                          MCValue &Target, uint64_t &Value,
                                          bool &IsResolved) {
  // The ADRP instruction adds some multiple of 0x1000 to the current PC &
  // ~0xfff. This means that the required offset to reach a symbol can vary by
  // up to one step depending on where the ADRP is in memory. For example:
  //
  //     ADRP x0, there
  //  there:
  //
  // If the ADRP occurs at address 0xffc then "there" will be at 0x1000 and
  // we'll need that as an offset. At any other address "there" will be in the
  // same page as the ADRP and the instruction should encode 0x0. Assuming the
  // section isn't 0x1000-aligned, we therefore need to delegate this decision
  // to the linker -- a relocation!
  if ((uint32_t)Fixup.getKind() == AArch64::fixup_a64_adr_prel_page ||
      (uint32_t)Fixup.getKind() == AArch64::fixup_a64_adr_prel_got_page ||
      (uint32_t)Fixup.getKind() == AArch64::fixup_a64_adr_gottprel_page ||
      (uint32_t)Fixup.getKind() == AArch64::fixup_a64_tlsdesc_adr_page)
    IsResolved = false;
}


static uint64_t adjustFixupValue(unsigned Kind, uint64_t Value);

namespace {

class ELFAArch64AsmBackend : public AArch64AsmBackend {
public:
  uint8_t OSABI;
  ELFAArch64AsmBackend(const Target &T, const StringRef TT,
                       uint8_t _OSABI)
    : AArch64AsmBackend(T, TT), OSABI(_OSABI) { }

  bool fixupNeedsRelaxation(const MCFixup &Fixup,
                            uint64_t Value,
                            const MCRelaxableFragment *DF,
                            const MCAsmLayout &Layout) const;

  unsigned int getNumFixupKinds() const {
    return AArch64::NumTargetFixupKinds;
  }

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const {
    const static MCFixupKindInfo Infos[AArch64::NumTargetFixupKinds] = {
// This table *must* be in the order that the fixup_* kinds are defined in
// AArch64FixupKinds.h.
//
// Name                   Offset (bits)    Size (bits)    Flags
{ "fixup_a64_ld_prel",               0,    32, MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_a64_adr_prel",              0,    32, MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_a64_adr_prel_page",         0,    32, MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_a64_add_lo12",              0,    32,             0 },
{ "fixup_a64_ldst8_lo12",            0,    32,             0 },
{ "fixup_a64_ldst16_lo12",           0,    32,             0 },
{ "fixup_a64_ldst32_lo12",           0,    32,             0 },
{ "fixup_a64_ldst64_lo12",           0,    32,             0 },
{ "fixup_a64_ldst128_lo12",          0,    32,             0 },
{ "fixup_a64_tstbr",                 0,    32, MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_a64_condbr",                0,    32, MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_a64_uncondbr",              0,    32, MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_a64_call",                  0,    32, MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_a64_movw_uabs_g0",          0,    32,             0 },
{ "fixup_a64_movw_uabs_g0_nc",       0,    32,             0 },
{ "fixup_a64_movw_uabs_g1",          0,    32,             0 },
{ "fixup_a64_movw_uabs_g1_nc",       0,    32,             0 },
{ "fixup_a64_movw_uabs_g2",          0,    32,             0 },
{ "fixup_a64_movw_uabs_g2_nc",       0,    32,             0 },
{ "fixup_a64_movw_uabs_g3",          0,    32,             0 },
{ "fixup_a64_movw_sabs_g0",          0,    32,             0 },
{ "fixup_a64_movw_sabs_g1",          0,    32,             0 },
{ "fixup_a64_movw_sabs_g2",          0,    32,             0 },
{ "fixup_a64_adr_prel_got_page",     0,    32, MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_a64_ld64_got_lo12_nc",      0,    32,             0 },
{ "fixup_a64_movw_dtprel_g2",        0,    32,             0 },
{ "fixup_a64_movw_dtprel_g1",        0,    32,             0 },
{ "fixup_a64_movw_dtprel_g1_nc",     0,    32,             0 },
{ "fixup_a64_movw_dtprel_g0",        0,    32,             0 },
{ "fixup_a64_movw_dtprel_g0_nc",     0,    32,             0 },
{ "fixup_a64_add_dtprel_hi12",       0,    32,             0 },
{ "fixup_a64_add_dtprel_lo12",       0,    32,             0 },
{ "fixup_a64_add_dtprel_lo12_nc",    0,    32,             0 },
{ "fixup_a64_ldst8_dtprel_lo12",     0,    32,             0 },
{ "fixup_a64_ldst8_dtprel_lo12_nc",  0,    32,             0 },
{ "fixup_a64_ldst16_dtprel_lo12",    0,    32,             0 },
{ "fixup_a64_ldst16_dtprel_lo12_nc", 0,    32,             0 },
{ "fixup_a64_ldst32_dtprel_lo12",    0,    32,             0 },
{ "fixup_a64_ldst32_dtprel_lo12_nc", 0,    32,             0 },
{ "fixup_a64_ldst64_dtprel_lo12",    0,    32,             0 },
{ "fixup_a64_ldst64_dtprel_lo12_nc", 0,    32,             0 },
{ "fixup_a64_movw_gottprel_g1",      0,    32,             0 },
{ "fixup_a64_movw_gottprel_g0_nc",   0,    32,             0 },
{ "fixup_a64_adr_gottprel_page",     0,    32, MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_a64_ld64_gottprel_lo12_nc", 0,    32,             0 },
{ "fixup_a64_ld_gottprel_prel19",    0,    32, MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_a64_movw_tprel_g2",         0,    32,             0 },
{ "fixup_a64_movw_tprel_g1",         0,    32,             0 },
{ "fixup_a64_movw_tprel_g1_nc",      0,    32,             0 },
{ "fixup_a64_movw_tprel_g0",         0,    32,             0 },
{ "fixup_a64_movw_tprel_g0_nc",      0,    32,             0 },
{ "fixup_a64_add_tprel_hi12",        0,    32,             0 },
{ "fixup_a64_add_tprel_lo12",        0,    32,             0 },
{ "fixup_a64_add_tprel_lo12_nc",     0,    32,             0 },
{ "fixup_a64_ldst8_tprel_lo12",      0,    32,             0 },
{ "fixup_a64_ldst8_tprel_lo12_nc",   0,    32,             0 },
{ "fixup_a64_ldst16_tprel_lo12",     0,    32,             0 },
{ "fixup_a64_ldst16_tprel_lo12_nc",  0,    32,             0 },
{ "fixup_a64_ldst32_tprel_lo12",     0,    32,             0 },
{ "fixup_a64_ldst32_tprel_lo12_nc",  0,    32,             0 },
{ "fixup_a64_ldst64_tprel_lo12",     0,    32,             0 },
{ "fixup_a64_ldst64_tprel_lo12_nc",  0,    32,             0 },
{ "fixup_a64_tlsdesc_adr_page",      0,    32, MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_a64_tlsdesc_ld64_lo12_nc",  0,    32,             0 },
{ "fixup_a64_tlsdesc_add_lo12_nc",   0,    32,             0 },
{ "fixup_a64_tlsdesc_call",          0,     0,             0 }
    };
    if (Kind < FirstTargetFixupKind)
      return MCAsmBackend::getFixupKindInfo(Kind);

    assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
           "Invalid kind!");
    return Infos[Kind - FirstTargetFixupKind];
  }

  void applyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                  uint64_t Value) const {
    unsigned NumBytes = getFixupKindInfo(Fixup.getKind()).TargetSize / 8;
    Value = adjustFixupValue(Fixup.getKind(), Value);
    if (!Value) return;           // Doesn't change encoding.

    unsigned Offset = Fixup.getOffset();
    assert(Offset + NumBytes <= DataSize && "Invalid fixup offset!");

    // For each byte of the fragment that the fixup touches, mask in the bits
    // from the fixup value.
    for (unsigned i = 0; i != NumBytes; ++i) {
      Data[Offset + i] |= uint8_t((Value >> (i * 8)) & 0xff);
    }
  }

  bool mayNeedRelaxation(const MCInst&) const {
    return false;
  }

  void relaxInstruction(const MCInst&, llvm::MCInst&) const {
    llvm_unreachable("Cannot relax instructions");
  }

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createAArch64ELFObjectWriter(OS, OSABI);
  }
};

} // end anonymous namespace

bool
ELFAArch64AsmBackend::fixupNeedsRelaxation(const MCFixup &Fixup,
                                           uint64_t Value,
                                           const MCRelaxableFragment *DF,
                                           const MCAsmLayout &Layout) const {
  // Correct for now. With all instructions 32-bit only very low-level
  // considerations could make you select something which may fail.
  return false;
}


bool AArch64AsmBackend::writeNopData(uint64_t Count, MCObjectWriter *OW) const {
  // Can't emit NOP with size not multiple of 32-bits
  if (Count % 4 != 0)
    return false;

  uint64_t NumNops = Count / 4;
  for (uint64_t i = 0; i != NumNops; ++i)
    OW->Write32(0xd503201f);

  return true;
}

static unsigned ADRImmBits(unsigned Value) {
  unsigned lo2 = Value & 0x3;
  unsigned hi19 = (Value & 0x1fffff) >> 2;

  return (hi19 << 5) | (lo2 << 29);
}

static uint64_t adjustFixupValue(unsigned Kind, uint64_t Value) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown fixup kind!");
  case FK_Data_2:
    assert((int64_t)Value >= -32768 &&
           (int64_t)Value <= 65536 &&
           "Out of range ABS16 fixup");
    return Value;
  case FK_Data_4:
    assert((int64_t)Value >= -(1LL << 31) &&
           (int64_t)Value <= (1LL << 32) - 1 &&
           "Out of range ABS32 fixup");
    return Value;
  case FK_Data_8:
    return Value;

  case AArch64::fixup_a64_ld_gottprel_prel19:
    // R_AARCH64_LD_GOTTPREL_PREL19: Set a load-literal immediate to bits 1F
    // FFFC of G(TPREL(S+A)) - P; check -2^20 <= X < 2^20.
  case AArch64::fixup_a64_ld_prel:
    // R_AARCH64_LD_PREL_LO19: Sets a load-literal (immediate) value to bits
    // 1F FFFC of S+A-P, checking that -2^20 <= S+A-P < 2^20.
    assert((int64_t)Value >= -(1LL << 20) &&
           (int64_t)Value < (1LL << 20) && "Out of range LDR (lit) fixup");
    return (Value & 0x1ffffc) << 3;

  case AArch64::fixup_a64_adr_prel:
    // R_AARCH64_ADR_PREL_LO21: Sets an ADR immediate value to bits 1F FFFF of
    // the result of S+A-P, checking that -2^20 <= S+A-P < 2^20.
    assert((int64_t)Value >= -(1LL << 20) &&
           (int64_t)Value < (1LL << 20) && "Out of range ADR fixup");
    return ADRImmBits(Value & 0x1fffff);

  case AArch64::fixup_a64_adr_prel_page:
    // R_AARCH64_ADR_PREL_PG_HI21: Sets an ADRP immediate value to bits 1 FFFF
    // F000 of the result of the operation, checking that -2^32 <= result <
    // 2^32.
    assert((int64_t)Value >= -(1LL << 32) &&
           (int64_t)Value < (1LL << 32) && "Out of range ADRP fixup");
    return ADRImmBits((Value & 0x1fffff000ULL) >> 12);

  case AArch64::fixup_a64_add_dtprel_hi12:
    // R_AARCH64_TLSLD_ADD_DTPREL_LO12: Set an ADD immediate field to bits
    // FF F000 of DTPREL(S+A), check 0 <= X < 2^24.
  case AArch64::fixup_a64_add_tprel_hi12:
    // R_AARCH64_TLSLD_ADD_TPREL_LO12: Set an ADD immediate field to bits
    // FF F000 of TPREL(S+A), check 0 <= X < 2^24.
    assert((int64_t)Value >= 0 &&
           (int64_t)Value < (1LL << 24) && "Out of range ADD fixup");
    return (Value & 0xfff000) >> 2;

  case AArch64::fixup_a64_add_dtprel_lo12:
    // R_AARCH64_TLSLD_ADD_DTPREL_LO12: Set an ADD immediate field to bits
    // FFF of DTPREL(S+A), check 0 <= X < 2^12.
  case AArch64::fixup_a64_add_tprel_lo12:
    // R_AARCH64_TLSLD_ADD_TPREL_LO12: Set an ADD immediate field to bits
    // FFF of TPREL(S+A), check 0 <= X < 2^12.
    assert((int64_t)Value >= 0 &&
           (int64_t)Value < (1LL << 12) && "Out of range ADD fixup");
    // ... fallthrough to no-checking versions ...
  case AArch64::fixup_a64_add_dtprel_lo12_nc:
    // R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC: Set an ADD immediate field to bits
    // FFF of DTPREL(S+A) with no overflow check.
  case AArch64::fixup_a64_add_tprel_lo12_nc:
    // R_AARCH64_TLSLD_ADD_TPREL_LO12_NC: Set an ADD immediate field to bits
    // FFF of TPREL(S+A) with no overflow check.
  case AArch64::fixup_a64_tlsdesc_add_lo12_nc:
    // R_AARCH64_TLSDESC_ADD_LO12_NC: Set an ADD immediate field to bits
    // FFF of G(TLSDESC(S+A)), with no overflow check.
  case AArch64::fixup_a64_add_lo12:
    // R_AARCH64_ADD_ABS_LO12_NC: Sets an ADD immediate value to bits FFF of
    // S+A, with no overflow check.
    return (Value & 0xfff) << 10;

  case AArch64::fixup_a64_ldst8_dtprel_lo12:
    // R_AARCH64_TLSLD_LDST8_DTPREL_LO12: Set an LD/ST offset field to bits FFF
    // of DTPREL(S+A), check 0 <= X < 2^12.
  case AArch64::fixup_a64_ldst8_tprel_lo12:
    // R_AARCH64_TLSLE_LDST8_TPREL_LO12: Set an LD/ST offset field to bits FFF
    // of DTPREL(S+A), check 0 <= X < 2^12.
    assert((int64_t) Value >= 0 &&
           (int64_t) Value < (1LL << 12) && "Out of range LD/ST fixup");
    // ... fallthrough to no-checking versions ...
  case AArch64::fixup_a64_ldst8_dtprel_lo12_nc:
    // R_AARCH64_TLSLD_LDST8_DTPREL_LO12: Set an LD/ST offset field to bits FFF
    // of DTPREL(S+A), with no overflow check.
  case AArch64::fixup_a64_ldst8_tprel_lo12_nc:
    // R_AARCH64_TLSLD_LDST8_TPREL_LO12: Set an LD/ST offset field to bits FFF
    // of TPREL(S+A), with no overflow check.
  case AArch64::fixup_a64_ldst8_lo12:
    // R_AARCH64_LDST8_ABS_LO12_NC: Sets an LD/ST immediate value to bits FFF
    // of S+A, with no overflow check.
    return (Value & 0xfff) << 10;

  case AArch64::fixup_a64_ldst16_dtprel_lo12:
    // R_AARCH64_TLSLD_LDST16_DTPREL_LO12: Set an LD/ST offset field to bits FFE
    // of DTPREL(S+A), check 0 <= X < 2^12.
  case AArch64::fixup_a64_ldst16_tprel_lo12:
    // R_AARCH64_TLSLE_LDST16_TPREL_LO12: Set an LD/ST offset field to bits FFE
    // of DTPREL(S+A), check 0 <= X < 2^12.
    assert((int64_t) Value >= 0 &&
           (int64_t) Value < (1LL << 12) && "Out of range LD/ST fixup");
    // ... fallthrough to no-checking versions ...
  case AArch64::fixup_a64_ldst16_dtprel_lo12_nc:
    // R_AARCH64_TLSLD_LDST16_DTPREL_LO12: Set an LD/ST offset field to bits FFE
    // of DTPREL(S+A), with no overflow check.
  case AArch64::fixup_a64_ldst16_tprel_lo12_nc:
    // R_AARCH64_TLSLD_LDST16_TPREL_LO12: Set an LD/ST offset field to bits FFE
    // of TPREL(S+A), with no overflow check.
  case AArch64::fixup_a64_ldst16_lo12:
    // R_AARCH64_LDST16_ABS_LO12_NC: Sets an LD/ST immediate value to bits FFE
    // of S+A, with no overflow check.
    return (Value & 0xffe) << 9;

  case AArch64::fixup_a64_ldst32_dtprel_lo12:
    // R_AARCH64_TLSLD_LDST32_DTPREL_LO12: Set an LD/ST offset field to bits FFC
    // of DTPREL(S+A), check 0 <= X < 2^12.
  case AArch64::fixup_a64_ldst32_tprel_lo12:
    // R_AARCH64_TLSLE_LDST32_TPREL_LO12: Set an LD/ST offset field to bits FFC
    // of DTPREL(S+A), check 0 <= X < 2^12.
    assert((int64_t) Value >= 0 &&
           (int64_t) Value < (1LL << 12) && "Out of range LD/ST fixup");
    // ... fallthrough to no-checking versions ...
  case AArch64::fixup_a64_ldst32_dtprel_lo12_nc:
    // R_AARCH64_TLSLD_LDST32_DTPREL_LO12: Set an LD/ST offset field to bits FFC
    // of DTPREL(S+A), with no overflow check.
  case AArch64::fixup_a64_ldst32_tprel_lo12_nc:
    // R_AARCH64_TLSLD_LDST32_TPREL_LO12: Set an LD/ST offset field to bits FFC
    // of TPREL(S+A), with no overflow check.
  case AArch64::fixup_a64_ldst32_lo12:
    // R_AARCH64_LDST32_ABS_LO12_NC: Sets an LD/ST immediate value to bits FFC
    // of S+A, with no overflow check.
    return (Value & 0xffc) << 8;

  case AArch64::fixup_a64_ldst64_dtprel_lo12:
    // R_AARCH64_TLSLD_LDST64_DTPREL_LO12: Set an LD/ST offset field to bits FF8
    // of DTPREL(S+A), check 0 <= X < 2^12.
  case AArch64::fixup_a64_ldst64_tprel_lo12:
    // R_AARCH64_TLSLE_LDST64_TPREL_LO12: Set an LD/ST offset field to bits FF8
    // of DTPREL(S+A), check 0 <= X < 2^12.
    assert((int64_t) Value >= 0 &&
           (int64_t) Value < (1LL << 12) && "Out of range LD/ST fixup");
    // ... fallthrough to no-checking versions ...
  case AArch64::fixup_a64_ldst64_dtprel_lo12_nc:
    // R_AARCH64_TLSLD_LDST64_DTPREL_LO12: Set an LD/ST offset field to bits FF8
    // of DTPREL(S+A), with no overflow check.
  case AArch64::fixup_a64_ldst64_tprel_lo12_nc:
    // R_AARCH64_TLSLD_LDST64_TPREL_LO12: Set an LD/ST offset field to bits FF8
    // of TPREL(S+A), with no overflow check.
  case AArch64::fixup_a64_ldst64_lo12:
    // R_AARCH64_LDST64_ABS_LO12_NC: Sets an LD/ST immediate value to bits FF8
    // of S+A, with no overflow check.
    return (Value & 0xff8) << 7;

  case AArch64::fixup_a64_ldst128_lo12:
    // R_AARCH64_LDST128_ABS_LO12_NC: Sets an LD/ST immediate value to bits FF0
    // of S+A, with no overflow check.
    return (Value & 0xff0) << 6;

  case AArch64::fixup_a64_movw_uabs_g0:
    // R_AARCH64_MOVW_UABS_G0: Sets a MOVZ immediate field to bits FFFF of S+A
    // with a check that S+A < 2^16
    assert(Value <= 0xffff && "Out of range move wide fixup");
    return (Value & 0xffff) << 5;

  case AArch64::fixup_a64_movw_dtprel_g0_nc:
    // R_AARCH64_TLSLD_MOVW_DTPREL_G0_NC: Sets a MOVK immediate field to bits
    // FFFF of DTPREL(S+A) with no overflow check.
  case AArch64::fixup_a64_movw_gottprel_g0_nc:
    // R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC: Sets a MOVK immediate field to bits
    // FFFF of G(TPREL(S+A)) - GOT with no overflow check.
  case AArch64::fixup_a64_movw_tprel_g0_nc:
    // R_AARCH64_TLSLE_MOVW_TPREL_G0_NC: Sets a MOVK immediate field to bits
    // FFFF of TPREL(S+A) with no overflow check.
  case AArch64::fixup_a64_movw_uabs_g0_nc:
    // R_AARCH64_MOVW_UABS_G0_NC: Sets a MOVK immediate field to bits FFFF of
    // S+A with no overflow check.
    return (Value & 0xffff) << 5;

  case AArch64::fixup_a64_movw_uabs_g1:
    // R_AARCH64_MOVW_UABS_G1: Sets a MOVZ immediate field to bits FFFF0000 of
    // S+A with a check that S+A < 2^32
    assert(Value <= 0xffffffffull && "Out of range move wide fixup");
    return ((Value >> 16) & 0xffff) << 5;

  case AArch64::fixup_a64_movw_dtprel_g1_nc:
    // R_AARCH64_TLSLD_MOVW_DTPREL_G1_NC: Set a MOVK immediate field
    // to bits FFFF0000 of DTPREL(S+A), with no overflow check.
  case AArch64::fixup_a64_movw_tprel_g1_nc:
    // R_AARCH64_TLSLD_MOVW_TPREL_G1_NC: Set a MOVK immediate field
    // to bits FFFF0000 of TPREL(S+A), with no overflow check.
  case AArch64::fixup_a64_movw_uabs_g1_nc:
    // R_AARCH64_MOVW_UABS_G1_NC: Sets a MOVK immediate field to bits
    // FFFF0000 of S+A with no overflow check.
    return ((Value >> 16) & 0xffff) << 5;

  case AArch64::fixup_a64_movw_uabs_g2:
    // R_AARCH64_MOVW_UABS_G2: Sets a MOVZ immediate field to bits FFFF 0000
    // 0000 of S+A with a check that S+A < 2^48
    assert(Value <= 0xffffffffffffull && "Out of range move wide fixup");
    return ((Value >> 32) & 0xffff) << 5;

  case AArch64::fixup_a64_movw_uabs_g2_nc:
    // R_AARCH64_MOVW_UABS_G2: Sets a MOVK immediate field to bits FFFF 0000
    // 0000 of S+A with no overflow check.
    return ((Value >> 32) & 0xffff) << 5;

  case AArch64::fixup_a64_movw_uabs_g3:
    // R_AARCH64_MOVW_UABS_G3: Sets a MOVZ immediate field to bits FFFF 0000
    // 0000 0000 of S+A (no overflow check needed)
    return ((Value >> 48) & 0xffff) << 5;

  case AArch64::fixup_a64_movw_dtprel_g0:
    // R_AARCH64_TLSLD_MOVW_DTPREL_G0: Set a MOV[NZ] immediate field
    // to bits FFFF of DTPREL(S+A).
  case AArch64::fixup_a64_movw_tprel_g0:
    // R_AARCH64_TLSLE_MOVW_TPREL_G0: Set a MOV[NZ] immediate field to
    // bits FFFF of TPREL(S+A).
  case AArch64::fixup_a64_movw_sabs_g0: {
    // R_AARCH64_MOVW_SABS_G0: Sets MOV[NZ] immediate field using bits FFFF of
    // S+A (see notes below); check -2^16 <= S+A < 2^16. (notes say that we
    // should convert between MOVN and MOVZ to achieve our goals).
    int64_t Signed = Value;
    assert(Signed >= -(1LL << 16) && Signed < (1LL << 16)
           && "Out of range move wide fixup");
    if (Signed >= 0) {
      Value = (Value & 0xffff) << 5;
      // Bit 30 converts the MOVN encoding into a MOVZ
      Value |= 1 << 30;
    } else {
      // MCCodeEmitter should have encoded a MOVN, which is fine.
      Value = (~Value & 0xffff) << 5;
    }
    return Value;
  }

  case AArch64::fixup_a64_movw_dtprel_g1:
    // R_AARCH64_TLSLD_MOVW_DTPREL_G1: Set a MOV[NZ] immediate field
    // to bits FFFF0000 of DTPREL(S+A).
  case AArch64::fixup_a64_movw_gottprel_g1:
    // R_AARCH64_TLSIE_MOVW_GOTTPREL_G1: Set a MOV[NZ] immediate field
    // to bits FFFF0000 of G(TPREL(S+A)) - GOT.
  case AArch64::fixup_a64_movw_tprel_g1:
    // R_AARCH64_TLSLE_MOVW_TPREL_G1: Set a MOV[NZ] immediate field to
    // bits FFFF0000 of TPREL(S+A).
  case AArch64::fixup_a64_movw_sabs_g1: {
    // R_AARCH64_MOVW_SABS_G1: Sets MOV[NZ] immediate field using bits FFFF 0000
    // of S+A (see notes below); check -2^32 <= S+A < 2^32. (notes say that we
    // should convert between MOVN and MOVZ to achieve our goals).
    int64_t Signed = Value;
    assert(Signed >= -(1LL << 32) && Signed < (1LL << 32)
           && "Out of range move wide fixup");
    if (Signed >= 0) {
      Value = ((Value >> 16) & 0xffff) << 5;
      // Bit 30 converts the MOVN encoding into a MOVZ
      Value |= 1 << 30;
    } else {
      Value = ((~Value >> 16) & 0xffff) << 5;
    }
    return Value;
  }

  case AArch64::fixup_a64_movw_dtprel_g2:
    // R_AARCH64_TLSLD_MOVW_DTPREL_G2: Set a MOV[NZ] immediate field
    // to bits FFFF 0000 0000 of DTPREL(S+A).
  case AArch64::fixup_a64_movw_tprel_g2:
    // R_AARCH64_TLSLE_MOVW_TPREL_G2: Set a MOV[NZ] immediate field to
    // bits FFFF 0000 0000 of TPREL(S+A).
  case AArch64::fixup_a64_movw_sabs_g2: {
    // R_AARCH64_MOVW_SABS_G2: Sets MOV[NZ] immediate field using bits FFFF 0000
    // 0000 of S+A (see notes below); check -2^48 <= S+A < 2^48. (notes say that
    // we should convert between MOVN and MOVZ to achieve our goals).
    int64_t Signed = Value;
    assert(Signed >= -(1LL << 48) && Signed < (1LL << 48)
           && "Out of range move wide fixup");
    if (Signed >= 0) {
      Value = ((Value >> 32) & 0xffff) << 5;
      // Bit 30 converts the MOVN encoding into a MOVZ
      Value |= 1 << 30;
    } else {
      Value = ((~Value >> 32) & 0xffff) << 5;
    }
    return Value;
  }

  case AArch64::fixup_a64_tstbr:
    // R_AARCH64_TSTBR14: Sets the immediate field of a TBZ/TBNZ instruction to
    // bits FFFC of S+A-P, checking -2^15 <= S+A-P < 2^15.
    assert((int64_t)Value >= -(1LL << 15) &&
           (int64_t)Value < (1LL << 15) && "Out of range TBZ/TBNZ fixup");
    return (Value & 0xfffc) << (5 - 2);

  case AArch64::fixup_a64_condbr:
    // R_AARCH64_CONDBR19: Sets the immediate field of a conditional branch
    // instruction to bits 1FFFFC of S+A-P, checking -2^20 <= S+A-P < 2^20.
    assert((int64_t)Value >= -(1LL << 20) &&
           (int64_t)Value < (1LL << 20) && "Out of range B.cond fixup");
    return (Value & 0x1ffffc) << (5 - 2);

  case AArch64::fixup_a64_uncondbr:
    // R_AARCH64_JUMP26 same as below (except to a linker, possibly).
  case AArch64::fixup_a64_call:
    // R_AARCH64_CALL26: Sets a CALL immediate field to bits FFFFFFC of S+A-P,
    // checking that -2^27 <= S+A-P < 2^27.
    assert((int64_t)Value >= -(1LL << 27) &&
           (int64_t)Value < (1LL << 27) && "Out of range branch fixup");
    return (Value & 0xffffffc) >> 2;

  case AArch64::fixup_a64_adr_gottprel_page:
    // R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21: Set an ADRP immediate field to bits
    // 1FFFFF000 of Page(G(TPREL(S+A))) - Page(P); check -2^32 <= X < 2^32.
  case AArch64::fixup_a64_tlsdesc_adr_page:
    // R_AARCH64_TLSDESC_ADR_PAGE: Set an ADRP immediate field to bits 1FFFFF000
    // of Page(G(TLSDESC(S+A))) - Page(P); check -2^32 <= X < 2^32.
  case AArch64::fixup_a64_adr_prel_got_page:
    // R_AARCH64_ADR_GOT_PAGE: Sets the immediate value of an ADRP to bits
    // 1FFFFF000 of the operation, checking that -2^32 < Page(G(S))-Page(GOT) <
    // 2^32.
    assert((int64_t)Value >= -(1LL << 32) &&
           (int64_t)Value < (1LL << 32) && "Out of range ADRP fixup");
    return ADRImmBits((Value & 0x1fffff000ULL) >> 12);

  case AArch64::fixup_a64_ld64_gottprel_lo12_nc:
    // R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC: Set an LD offset field to bits FF8
    // of X, with no overflow check. Check that X & 7 == 0.
  case AArch64::fixup_a64_tlsdesc_ld64_lo12_nc:
    // R_AARCH64_TLSDESC_LD64_LO12_NC: Set an LD offset field to bits FF8 of
    // G(TLSDESC(S+A)), with no overflow check. Check that X & 7 == 0.
  case AArch64::fixup_a64_ld64_got_lo12_nc:
    // R_AARCH64_LD64_GOT_LO12_NC: Sets the LD/ST immediate field to bits FF8 of
    // G(S) with no overflow check. Check X & 7 == 0
    assert(((int64_t)Value & 7) == 0 && "Misaligned fixup");
    return (Value & 0xff8) << 7;

  case AArch64::fixup_a64_tlsdesc_call:
    // R_AARCH64_TLSDESC_CALL: For relaxation only.
    return 0;
  }
}

MCAsmBackend *
llvm::createAArch64AsmBackend(const Target &T, const MCRegisterInfo &MRI,
                              StringRef TT, StringRef CPU) {
  Triple TheTriple(TT);
  return new ELFAArch64AsmBackend(T, TT, TheTriple.getOS());
}
