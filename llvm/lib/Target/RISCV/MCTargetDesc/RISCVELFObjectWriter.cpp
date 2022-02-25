//===-- RISCVELFObjectWriter.cpp - RISCV ELF Writer -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVFixupKinds.h"
#include "MCTargetDesc/RISCVMCExpr.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
class RISCVELFObjectWriter : public MCELFObjectTargetWriter {
public:
  RISCVELFObjectWriter(uint8_t OSABI, bool Is64Bit);

  ~RISCVELFObjectWriter() override;

  // Return true if the given relocation must be with a symbol rather than
  // section plus offset.
  bool needsRelocateWithSymbol(const MCSymbol &Sym,
                               unsigned Type) const override {
    // TODO: this is very conservative, update once RISC-V psABI requirements
    //       are clarified.
    return true;
  }

protected:
  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const override;
};
}

RISCVELFObjectWriter::RISCVELFObjectWriter(uint8_t OSABI, bool Is64Bit)
    : MCELFObjectTargetWriter(Is64Bit, OSABI, ELF::EM_RISCV,
                              /*HasRelocationAddend*/ true) {}

RISCVELFObjectWriter::~RISCVELFObjectWriter() = default;

unsigned RISCVELFObjectWriter::getRelocType(MCContext &Ctx,
                                            const MCValue &Target,
                                            const MCFixup &Fixup,
                                            bool IsPCRel) const {
  const MCExpr *Expr = Fixup.getValue();
  // Determine the type of the relocation
  unsigned Kind = Fixup.getTargetKind();
  if (Kind >= FirstLiteralRelocationKind)
    return Kind - FirstLiteralRelocationKind;
  if (IsPCRel) {
    switch (Kind) {
    default:
      Ctx.reportError(Fixup.getLoc(), "Unsupported relocation type");
      return ELF::R_RISCV_NONE;
    case FK_Data_4:
    case FK_PCRel_4:
      return ELF::R_RISCV_32_PCREL;
    case RISCV::fixup_riscv_pcrel_hi20:
      return ELF::R_RISCV_PCREL_HI20;
    case RISCV::fixup_riscv_pcrel_lo12_i:
      return ELF::R_RISCV_PCREL_LO12_I;
    case RISCV::fixup_riscv_pcrel_lo12_s:
      return ELF::R_RISCV_PCREL_LO12_S;
    case RISCV::fixup_riscv_got_hi20:
      return ELF::R_RISCV_GOT_HI20;
    case RISCV::fixup_riscv_tls_got_hi20:
      return ELF::R_RISCV_TLS_GOT_HI20;
    case RISCV::fixup_riscv_tls_gd_hi20:
      return ELF::R_RISCV_TLS_GD_HI20;
    case RISCV::fixup_riscv_jal:
      return ELF::R_RISCV_JAL;
    case RISCV::fixup_riscv_branch:
      return ELF::R_RISCV_BRANCH;
    case RISCV::fixup_riscv_rvc_jump:
      return ELF::R_RISCV_RVC_JUMP;
    case RISCV::fixup_riscv_rvc_branch:
      return ELF::R_RISCV_RVC_BRANCH;
    case RISCV::fixup_riscv_call:
      return ELF::R_RISCV_CALL;
    case RISCV::fixup_riscv_call_plt:
      return ELF::R_RISCV_CALL_PLT;
    case RISCV::fixup_riscv_add_8:
      return ELF::R_RISCV_ADD8;
    case RISCV::fixup_riscv_sub_8:
      return ELF::R_RISCV_SUB8;
    case RISCV::fixup_riscv_add_16:
      return ELF::R_RISCV_ADD16;
    case RISCV::fixup_riscv_sub_16:
      return ELF::R_RISCV_SUB16;
    case RISCV::fixup_riscv_add_32:
      return ELF::R_RISCV_ADD32;
    case RISCV::fixup_riscv_sub_32:
      return ELF::R_RISCV_SUB32;
    case RISCV::fixup_riscv_add_64:
      return ELF::R_RISCV_ADD64;
    case RISCV::fixup_riscv_sub_64:
      return ELF::R_RISCV_SUB64;
    }
  }

  switch (Kind) {
  default:
    Ctx.reportError(Fixup.getLoc(), "Unsupported relocation type");
    return ELF::R_RISCV_NONE;
  case FK_Data_1:
    Ctx.reportError(Fixup.getLoc(), "1-byte data relocations not supported");
    return ELF::R_RISCV_NONE;
  case FK_Data_2:
    Ctx.reportError(Fixup.getLoc(), "2-byte data relocations not supported");
    return ELF::R_RISCV_NONE;
  case FK_Data_4:
    if (Expr->getKind() == MCExpr::Target &&
        cast<RISCVMCExpr>(Expr)->getKind() == RISCVMCExpr::VK_RISCV_32_PCREL)
      return ELF::R_RISCV_32_PCREL;
    return ELF::R_RISCV_32;
  case FK_Data_8:
    return ELF::R_RISCV_64;
  case RISCV::fixup_riscv_hi20:
    return ELF::R_RISCV_HI20;
  case RISCV::fixup_riscv_lo12_i:
    return ELF::R_RISCV_LO12_I;
  case RISCV::fixup_riscv_lo12_s:
    return ELF::R_RISCV_LO12_S;
  case RISCV::fixup_riscv_tprel_hi20:
    return ELF::R_RISCV_TPREL_HI20;
  case RISCV::fixup_riscv_tprel_lo12_i:
    return ELF::R_RISCV_TPREL_LO12_I;
  case RISCV::fixup_riscv_tprel_lo12_s:
    return ELF::R_RISCV_TPREL_LO12_S;
  case RISCV::fixup_riscv_tprel_add:
    return ELF::R_RISCV_TPREL_ADD;
  case RISCV::fixup_riscv_relax:
    return ELF::R_RISCV_RELAX;
  case RISCV::fixup_riscv_align:
    return ELF::R_RISCV_ALIGN;
  case RISCV::fixup_riscv_set_6b:
    return ELF::R_RISCV_SET6;
  case RISCV::fixup_riscv_sub_6b:
    return ELF::R_RISCV_SUB6;
  case RISCV::fixup_riscv_add_8:
    return ELF::R_RISCV_ADD8;
  case RISCV::fixup_riscv_set_8:
    return ELF::R_RISCV_SET8;
  case RISCV::fixup_riscv_sub_8:
    return ELF::R_RISCV_SUB8;
  case RISCV::fixup_riscv_set_16:
    return ELF::R_RISCV_SET16;
  case RISCV::fixup_riscv_add_16:
    return ELF::R_RISCV_ADD16;
  case RISCV::fixup_riscv_sub_16:
    return ELF::R_RISCV_SUB16;
  case RISCV::fixup_riscv_set_32:
    return ELF::R_RISCV_SET32;
  case RISCV::fixup_riscv_add_32:
    return ELF::R_RISCV_ADD32;
  case RISCV::fixup_riscv_sub_32:
    return ELF::R_RISCV_SUB32;
  case RISCV::fixup_riscv_add_64:
    return ELF::R_RISCV_ADD64;
  case RISCV::fixup_riscv_sub_64:
    return ELF::R_RISCV_SUB64;
  }
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createRISCVELFObjectWriter(uint8_t OSABI, bool Is64Bit) {
  return std::make_unique<RISCVELFObjectWriter>(OSABI, Is64Bit);
}
