//===-- SparcELFObjectWriter.cpp - Sparc ELF Writer -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SparcFixupKinds.h"
#include "MCTargetDesc/SparcMCExpr.h"
#include "MCTargetDesc/SparcMCTargetDesc.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
  class SparcELFObjectWriter : public MCELFObjectTargetWriter {
  public:
    SparcELFObjectWriter(bool Is64Bit, uint8_t OSABI)
      : MCELFObjectTargetWriter(Is64Bit, OSABI,
                                Is64Bit ?  ELF::EM_SPARCV9 : ELF::EM_SPARC,
                                /*HasRelocationAddend*/ true) {}

    virtual ~SparcELFObjectWriter() {}
  protected:
    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend) const;

    virtual const MCSymbol *ExplicitRelSym(const MCAssembler &Asm,
                                           const MCValue &Target,
                                           const MCFragment &F,
                                           const MCFixup &Fixup,
                                           bool IsPCRel) const;
  };
}


unsigned SparcELFObjectWriter::GetRelocType(const MCValue &Target,
                                            const MCFixup &Fixup,
                                            bool IsPCRel,
                                            bool IsRelocWithSymbol,
                                            int64_t Addend) const {

  if (const SparcMCExpr *SExpr = dyn_cast<SparcMCExpr>(Fixup.getValue())) {
    if (SExpr->getKind() == SparcMCExpr::VK_Sparc_R_DISP32)
      return ELF::R_SPARC_DISP32;
  }

  if (IsPCRel) {
    switch((unsigned)Fixup.getKind()) {
    default:
      llvm_unreachable("Unimplemented fixup -> relocation");
    case FK_Data_1:                  return ELF::R_SPARC_DISP8;
    case FK_Data_2:                  return ELF::R_SPARC_DISP16;
    case FK_Data_4:                  return ELF::R_SPARC_DISP32;
    case FK_Data_8:                  return ELF::R_SPARC_DISP64;
    case Sparc::fixup_sparc_call30:  return ELF::R_SPARC_WDISP30;
    case Sparc::fixup_sparc_br22:    return ELF::R_SPARC_WDISP22;
    case Sparc::fixup_sparc_br19:    return ELF::R_SPARC_WDISP19;
    case Sparc::fixup_sparc_pc22:    return ELF::R_SPARC_PC22;
    case Sparc::fixup_sparc_pc10:    return ELF::R_SPARC_PC10;
    case Sparc::fixup_sparc_wplt30:  return ELF::R_SPARC_WPLT30;
    }
  }

  switch((unsigned)Fixup.getKind()) {
  default:
    llvm_unreachable("Unimplemented fixup -> relocation");
  case FK_Data_1:                return ELF::R_SPARC_8;
  case FK_Data_2:                return ((Fixup.getOffset() % 2)
                                         ? ELF::R_SPARC_UA16
                                         : ELF::R_SPARC_16);
  case FK_Data_4:                return ((Fixup.getOffset() % 4)
                                         ? ELF::R_SPARC_UA32
                                         : ELF::R_SPARC_32);
  case FK_Data_8:                return ((Fixup.getOffset() % 8)
                                         ? ELF::R_SPARC_UA64
                                         : ELF::R_SPARC_64);
  case Sparc::fixup_sparc_hi22:  return ELF::R_SPARC_HI22;
  case Sparc::fixup_sparc_lo10:  return ELF::R_SPARC_LO10;
  case Sparc::fixup_sparc_h44:   return ELF::R_SPARC_H44;
  case Sparc::fixup_sparc_m44:   return ELF::R_SPARC_M44;
  case Sparc::fixup_sparc_l44:   return ELF::R_SPARC_L44;
  case Sparc::fixup_sparc_hh:    return ELF::R_SPARC_HH22;
  case Sparc::fixup_sparc_hm:    return ELF::R_SPARC_HM10;
  case Sparc::fixup_sparc_got22: return ELF::R_SPARC_GOT22;
  case Sparc::fixup_sparc_got10: return ELF::R_SPARC_GOT10;
  }

  return ELF::R_SPARC_NONE;
}

const MCSymbol *SparcELFObjectWriter::ExplicitRelSym(const MCAssembler &Asm,
                                                     const MCValue &Target,
                                                     const MCFragment &F,
                                                     const MCFixup &Fixup,
                                                     bool IsPCRel) const {

  if (!Target.getSymA())
    return NULL;
  switch((unsigned)Fixup.getKind()) {
  default: break;
  case Sparc::fixup_sparc_got22:
  case Sparc::fixup_sparc_got10:
    return &Target.getSymA()->getSymbol().AliasedSymbol();
  }
  return NULL;
}

MCObjectWriter *llvm::createSparcELFObjectWriter(raw_ostream &OS,
                                                 bool Is64Bit,
                                                 uint8_t OSABI) {
  MCELFObjectTargetWriter *MOTW = new SparcELFObjectWriter(Is64Bit, OSABI);
  return createELFObjectWriter(MOTW, OS,  /*IsLittleEndian=*/false);
}
