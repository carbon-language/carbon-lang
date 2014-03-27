//===-- AArch64ELFObjectWriter.cpp - AArch64 ELF Writer -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file handles ELF-specific object emission, converting LLVM's internal
// fixups into the appropriate relocations.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AArch64FixupKinds.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
class AArch64ELFObjectWriter : public MCELFObjectTargetWriter {
public:
  AArch64ELFObjectWriter(uint8_t OSABI, bool IsLittleEndian);

  virtual ~AArch64ELFObjectWriter();

protected:
  unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                        bool IsPCRel, bool IsRelocWithSymbol) const override;

private:
};
}

AArch64ELFObjectWriter::AArch64ELFObjectWriter(uint8_t OSABI, bool IsLittleEndian)
  : MCELFObjectTargetWriter(/*Is64Bit*/ true, OSABI, ELF::EM_AARCH64,
                            /*HasRelocationAddend*/ true)
{}

AArch64ELFObjectWriter::~AArch64ELFObjectWriter()
{}

unsigned AArch64ELFObjectWriter::GetRelocType(const MCValue &Target,
                                              const MCFixup &Fixup,
                                              bool IsPCRel,
                                              bool IsRelocWithSymbol) const {
  unsigned Type;
  if (IsPCRel) {
    switch ((unsigned)Fixup.getKind()) {
    default:
      llvm_unreachable("Unimplemented fixup -> relocation");
    case FK_Data_8:
      return ELF::R_AARCH64_PREL64;
    case FK_Data_4:
      return ELF::R_AARCH64_PREL32;
    case FK_Data_2:
      return ELF::R_AARCH64_PREL16;
    case AArch64::fixup_a64_ld_prel:
      Type = ELF::R_AARCH64_LD_PREL_LO19;
      break;
    case AArch64::fixup_a64_adr_prel:
      Type = ELF::R_AARCH64_ADR_PREL_LO21;
      break;
    case AArch64::fixup_a64_adr_prel_page:
      Type = ELF::R_AARCH64_ADR_PREL_PG_HI21;
      break;
    case AArch64::fixup_a64_adr_prel_got_page:
      Type = ELF::R_AARCH64_ADR_GOT_PAGE;
      break;
    case AArch64::fixup_a64_tstbr:
      Type = ELF::R_AARCH64_TSTBR14;
      break;
    case AArch64::fixup_a64_condbr:
      Type = ELF::R_AARCH64_CONDBR19;
      break;
    case AArch64::fixup_a64_uncondbr:
      Type = ELF::R_AARCH64_JUMP26;
      break;
    case AArch64::fixup_a64_call:
      Type = ELF::R_AARCH64_CALL26;
      break;
    case AArch64::fixup_a64_adr_gottprel_page:
      Type = ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21;
      break;
    case AArch64::fixup_a64_ld_gottprel_prel19:
      Type =  ELF::R_AARCH64_TLSIE_LD_GOTTPREL_PREL19;
      break;
    case AArch64::fixup_a64_tlsdesc_adr_page:
      Type = ELF::R_AARCH64_TLSDESC_ADR_PAGE;
      break;
    }
  } else {
    switch ((unsigned)Fixup.getKind()) {
    default:
      llvm_unreachable("Unimplemented fixup -> relocation");
    case FK_Data_8:
      return ELF::R_AARCH64_ABS64;
    case FK_Data_4:
      return ELF::R_AARCH64_ABS32;
    case FK_Data_2:
      return ELF::R_AARCH64_ABS16;
    case AArch64::fixup_a64_add_lo12:
      Type = ELF::R_AARCH64_ADD_ABS_LO12_NC;
      break;
    case AArch64::fixup_a64_ld64_got_lo12_nc:
      Type = ELF::R_AARCH64_LD64_GOT_LO12_NC;
      break;
    case AArch64::fixup_a64_ldst8_lo12:
      Type = ELF::R_AARCH64_LDST8_ABS_LO12_NC;
      break;
    case AArch64::fixup_a64_ldst16_lo12:
      Type = ELF::R_AARCH64_LDST16_ABS_LO12_NC;
      break;
    case AArch64::fixup_a64_ldst32_lo12:
      Type = ELF::R_AARCH64_LDST32_ABS_LO12_NC;
      break;
    case AArch64::fixup_a64_ldst64_lo12:
      Type = ELF::R_AARCH64_LDST64_ABS_LO12_NC;
      break;
    case AArch64::fixup_a64_ldst128_lo12:
      Type = ELF::R_AARCH64_LDST128_ABS_LO12_NC;
      break;
    case AArch64::fixup_a64_movw_uabs_g0:
      Type = ELF::R_AARCH64_MOVW_UABS_G0;
      break;
    case AArch64::fixup_a64_movw_uabs_g0_nc:
      Type = ELF::R_AARCH64_MOVW_UABS_G0_NC;
      break;
    case AArch64::fixup_a64_movw_uabs_g1:
      Type = ELF::R_AARCH64_MOVW_UABS_G1;
      break;
    case AArch64::fixup_a64_movw_uabs_g1_nc:
      Type = ELF::R_AARCH64_MOVW_UABS_G1_NC;
      break;
    case AArch64::fixup_a64_movw_uabs_g2:
      Type = ELF::R_AARCH64_MOVW_UABS_G2;
      break;
    case AArch64::fixup_a64_movw_uabs_g2_nc:
      Type = ELF::R_AARCH64_MOVW_UABS_G2_NC;
      break;
    case AArch64::fixup_a64_movw_uabs_g3:
      Type = ELF::R_AARCH64_MOVW_UABS_G3;
      break;
    case AArch64::fixup_a64_movw_sabs_g0:
      Type = ELF::R_AARCH64_MOVW_SABS_G0;
      break;
    case AArch64::fixup_a64_movw_sabs_g1:
      Type = ELF::R_AARCH64_MOVW_SABS_G1;
      break;
    case AArch64::fixup_a64_movw_sabs_g2:
      Type = ELF::R_AARCH64_MOVW_SABS_G2;
      break;

    // TLS Local-dynamic block
    case AArch64::fixup_a64_movw_dtprel_g2:
      Type = ELF::R_AARCH64_TLSLD_MOVW_DTPREL_G2;
      break;
    case AArch64::fixup_a64_movw_dtprel_g1:
      Type = ELF::R_AARCH64_TLSLD_MOVW_DTPREL_G1;
      break;
    case AArch64::fixup_a64_movw_dtprel_g1_nc:
      Type = ELF::R_AARCH64_TLSLD_MOVW_DTPREL_G1_NC;
      break;
    case AArch64::fixup_a64_movw_dtprel_g0:
      Type = ELF::R_AARCH64_TLSLD_MOVW_DTPREL_G0;
      break;
    case AArch64::fixup_a64_movw_dtprel_g0_nc:
      Type = ELF::R_AARCH64_TLSLD_MOVW_DTPREL_G0_NC;
      break;
    case AArch64::fixup_a64_add_dtprel_hi12:
      Type = ELF::R_AARCH64_TLSLD_ADD_DTPREL_HI12;
      break;
    case AArch64::fixup_a64_add_dtprel_lo12:
      Type = ELF::R_AARCH64_TLSLD_ADD_DTPREL_LO12;
      break;
    case AArch64::fixup_a64_add_dtprel_lo12_nc:
      Type = ELF::R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC;
      break;
    case AArch64::fixup_a64_ldst8_dtprel_lo12:
      Type = ELF::R_AARCH64_TLSLD_LDST8_DTPREL_LO12;
      break;
    case AArch64::fixup_a64_ldst8_dtprel_lo12_nc:
      Type = ELF::R_AARCH64_TLSLD_LDST8_DTPREL_LO12_NC;
      break;
    case AArch64::fixup_a64_ldst16_dtprel_lo12:
      Type = ELF::R_AARCH64_TLSLD_LDST16_DTPREL_LO12;
      break;
    case AArch64::fixup_a64_ldst16_dtprel_lo12_nc:
      Type = ELF::R_AARCH64_TLSLD_LDST16_DTPREL_LO12_NC;
      break;
    case AArch64::fixup_a64_ldst32_dtprel_lo12:
      Type = ELF::R_AARCH64_TLSLD_LDST32_DTPREL_LO12;
      break;
    case AArch64::fixup_a64_ldst32_dtprel_lo12_nc:
      Type = ELF::R_AARCH64_TLSLD_LDST32_DTPREL_LO12_NC;
      break;
    case AArch64::fixup_a64_ldst64_dtprel_lo12:
      Type = ELF::R_AARCH64_TLSLD_LDST64_DTPREL_LO12;
      break;
    case AArch64::fixup_a64_ldst64_dtprel_lo12_nc:
      Type = ELF::R_AARCH64_TLSLD_LDST64_DTPREL_LO12_NC;
      break;

    // TLS initial-exec block
    case AArch64::fixup_a64_movw_gottprel_g1:
      Type = ELF::R_AARCH64_TLSIE_MOVW_GOTTPREL_G1;
      break;
    case AArch64::fixup_a64_movw_gottprel_g0_nc:
      Type = ELF::R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC;
      break;
    case AArch64::fixup_a64_ld64_gottprel_lo12_nc:
      Type = ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC;
      break;

    // TLS local-exec block
    case AArch64::fixup_a64_movw_tprel_g2:
      Type = ELF::R_AARCH64_TLSLE_MOVW_TPREL_G2;
      break;
    case AArch64::fixup_a64_movw_tprel_g1:
      Type = ELF::R_AARCH64_TLSLE_MOVW_TPREL_G1;
      break;
    case AArch64::fixup_a64_movw_tprel_g1_nc:
      Type = ELF::R_AARCH64_TLSLE_MOVW_TPREL_G1_NC;
      break;
    case AArch64::fixup_a64_movw_tprel_g0:
      Type = ELF::R_AARCH64_TLSLE_MOVW_TPREL_G0;
      break;
    case AArch64::fixup_a64_movw_tprel_g0_nc:
      Type = ELF::R_AARCH64_TLSLE_MOVW_TPREL_G0_NC;
      break;
    case AArch64::fixup_a64_add_tprel_hi12:
      Type = ELF::R_AARCH64_TLSLE_ADD_TPREL_HI12;
      break;
    case AArch64::fixup_a64_add_tprel_lo12:
      Type = ELF::R_AARCH64_TLSLE_ADD_TPREL_LO12;
      break;
    case AArch64::fixup_a64_add_tprel_lo12_nc:
      Type = ELF::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC;
      break;
    case AArch64::fixup_a64_ldst8_tprel_lo12:
      Type = ELF::R_AARCH64_TLSLE_LDST8_TPREL_LO12;
      break;
    case AArch64::fixup_a64_ldst8_tprel_lo12_nc:
      Type = ELF::R_AARCH64_TLSLE_LDST8_TPREL_LO12_NC;
      break;
    case AArch64::fixup_a64_ldst16_tprel_lo12:
      Type = ELF::R_AARCH64_TLSLE_LDST16_TPREL_LO12;
      break;
    case AArch64::fixup_a64_ldst16_tprel_lo12_nc:
      Type = ELF::R_AARCH64_TLSLE_LDST16_TPREL_LO12_NC;
      break;
    case AArch64::fixup_a64_ldst32_tprel_lo12:
      Type = ELF::R_AARCH64_TLSLE_LDST32_TPREL_LO12;
      break;
    case AArch64::fixup_a64_ldst32_tprel_lo12_nc:
      Type = ELF::R_AARCH64_TLSLE_LDST32_TPREL_LO12_NC;
      break;
    case AArch64::fixup_a64_ldst64_tprel_lo12:
      Type = ELF::R_AARCH64_TLSLE_LDST64_TPREL_LO12;
      break;
    case AArch64::fixup_a64_ldst64_tprel_lo12_nc:
      Type = ELF::R_AARCH64_TLSLE_LDST64_TPREL_LO12_NC;
      break;

    // TLS general-dynamic block
    case AArch64::fixup_a64_tlsdesc_adr_page:
      Type = ELF::R_AARCH64_TLSDESC_ADR_PAGE;
      break;
    case AArch64::fixup_a64_tlsdesc_ld64_lo12_nc:
      Type = ELF::R_AARCH64_TLSDESC_LD64_LO12_NC;
      break;
    case AArch64::fixup_a64_tlsdesc_add_lo12_nc:
      Type = ELF::R_AARCH64_TLSDESC_ADD_LO12_NC;
      break;
    case AArch64::fixup_a64_tlsdesc_call:
      Type = ELF::R_AARCH64_TLSDESC_CALL;
      break;
    }
  }

  return Type;
}

MCObjectWriter *llvm::createAArch64ELFObjectWriter(raw_ostream &OS,
                                                   uint8_t OSABI,
                                                   bool IsLittleEndian) {
  MCELFObjectTargetWriter *MOTW = new AArch64ELFObjectWriter(OSABI, IsLittleEndian);
  return createELFObjectWriter(MOTW, OS,  IsLittleEndian);
}
