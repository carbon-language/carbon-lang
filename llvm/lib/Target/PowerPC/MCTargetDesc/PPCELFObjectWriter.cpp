//===-- PPCELFObjectWriter.cpp - PPC ELF Writer ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PPCMCTargetDesc.h"
#include "MCTargetDesc/PPCFixupKinds.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
  class PPCELFObjectWriter : public MCELFObjectTargetWriter {
  public:
    PPCELFObjectWriter(bool Is64Bit, uint8_t OSABI);

    virtual ~PPCELFObjectWriter();
  protected:
    virtual unsigned getRelocTypeInner(const MCValue &Target,
                                       const MCFixup &Fixup,
                                       bool IsPCRel) const;
    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend) const;
    virtual const MCSymbol *undefinedExplicitRelSym(const MCValue &Target,
                                                    const MCFixup &Fixup,
                                                    bool IsPCRel) const;
  };
}

PPCELFObjectWriter::PPCELFObjectWriter(bool Is64Bit, uint8_t OSABI)
  : MCELFObjectTargetWriter(Is64Bit, OSABI,
                            Is64Bit ?  ELF::EM_PPC64 : ELF::EM_PPC,
                            /*HasRelocationAddend*/ true) {}

PPCELFObjectWriter::~PPCELFObjectWriter() {
}

unsigned PPCELFObjectWriter::getRelocTypeInner(const MCValue &Target,
                                               const MCFixup &Fixup,
                                               bool IsPCRel) const
{
  MCSymbolRefExpr::VariantKind Modifier = Target.isAbsolute() ?
    MCSymbolRefExpr::VK_None : Target.getSymA()->getKind();

  // determine the type of the relocation
  unsigned Type;
  if (IsPCRel) {
    switch ((unsigned)Fixup.getKind()) {
    default:
      llvm_unreachable("Unimplemented");
    case PPC::fixup_ppc_br24:
      Type = ELF::R_PPC_REL24;
      break;
    case PPC::fixup_ppc_brcond14:
      Type = ELF::R_PPC_REL14;
      break;
    case FK_Data_4:
    case FK_PCRel_4:
      Type = ELF::R_PPC_REL32;
      break;
    case FK_Data_8:
    case FK_PCRel_8:
      Type = ELF::R_PPC64_REL64;
      break;
    }
  } else {
    switch ((unsigned)Fixup.getKind()) {
      default: llvm_unreachable("invalid fixup kind!");
    case PPC::fixup_ppc_br24:
      Type = ELF::R_PPC_ADDR24;
      break;
    case PPC::fixup_ppc_brcond14:
      Type = ELF::R_PPC_ADDR14; // XXX: or BRNTAKEN?_
      break;
    case PPC::fixup_ppc_half16:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier");
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_PPC_ADDR16;
        break;
      case MCSymbolRefExpr::VK_PPC_LO:
        Type = ELF::R_PPC_ADDR16_LO;
        break;
      case MCSymbolRefExpr::VK_PPC_HI:
        Type = ELF::R_PPC_ADDR16_HI;
        break;
      case MCSymbolRefExpr::VK_PPC_HA:
        Type = ELF::R_PPC_ADDR16_HA;
        break;
      case MCSymbolRefExpr::VK_PPC_TOC:
        Type = ELF::R_PPC64_TOC16;
        break;
      case MCSymbolRefExpr::VK_PPC_TOC_LO:
        Type = ELF::R_PPC64_TOC16_LO;
        break;
      case MCSymbolRefExpr::VK_PPC_TOC_HI:
        Type = ELF::R_PPC64_TOC16_HI;
        break;
      case MCSymbolRefExpr::VK_PPC_TOC_HA:
        Type = ELF::R_PPC64_TOC16_HA;
        break;
      case MCSymbolRefExpr::VK_PPC_TPREL_LO:
        Type = ELF::R_PPC_TPREL16_LO;
        break;
      case MCSymbolRefExpr::VK_PPC_TPREL_HA:
        Type = ELF::R_PPC_TPREL16_HA;
        break;
      case MCSymbolRefExpr::VK_PPC_DTPREL_LO:
        Type = ELF::R_PPC64_DTPREL16_LO;
        break;
      case MCSymbolRefExpr::VK_PPC_DTPREL_HA:
        Type = ELF::R_PPC64_DTPREL16_HA;
        break;
      case MCSymbolRefExpr::VK_PPC_GOT_TLSGD_LO:
        Type = ELF::R_PPC64_GOT_TLSGD16_LO;
        break;
      case MCSymbolRefExpr::VK_PPC_GOT_TLSGD_HA:
        Type = ELF::R_PPC64_GOT_TLSGD16_HA;
        break;
      case MCSymbolRefExpr::VK_PPC_GOT_TLSLD_LO:
        Type = ELF::R_PPC64_GOT_TLSLD16_LO;
        break;
      case MCSymbolRefExpr::VK_PPC_GOT_TLSLD_HA:
        Type = ELF::R_PPC64_GOT_TLSLD16_HA;
        break;
      case MCSymbolRefExpr::VK_PPC_GOT_TPREL_HA:
        Type = ELF::R_PPC64_GOT_TPREL16_HA;
        break;
      }
      break;
    case PPC::fixup_ppc_half16ds:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier");
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_PPC64_ADDR16_DS;
        break;
      case MCSymbolRefExpr::VK_PPC_LO:
        Type = ELF::R_PPC64_ADDR16_LO_DS;
        break;
      case MCSymbolRefExpr::VK_PPC_TOC:
        Type = ELF::R_PPC64_TOC16_DS;
	break;
      case MCSymbolRefExpr::VK_PPC_TOC_LO:
        Type = ELF::R_PPC64_TOC16_LO_DS;
        break;
      case MCSymbolRefExpr::VK_PPC_GOT_TPREL_LO:
        Type = ELF::R_PPC64_GOT_TPREL16_LO_DS;
        break;
      }
      break;
    case PPC::fixup_ppc_tlsreg:
      Type = ELF::R_PPC64_TLS;
      break;
    case PPC::fixup_ppc_nofixup:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier");
      case MCSymbolRefExpr::VK_PPC_TLSGD:
        Type = ELF::R_PPC64_TLSGD;
        break;
      case MCSymbolRefExpr::VK_PPC_TLSLD:
        Type = ELF::R_PPC64_TLSLD;
        break;
      }
      break;
    case FK_Data_8:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier");
      case MCSymbolRefExpr::VK_PPC_TOCBASE:
        Type = ELF::R_PPC64_TOC;
        break;
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_PPC64_ADDR64;
	break;
      }
      break;
    case FK_Data_4:
      Type = ELF::R_PPC_ADDR32;
      break;
    case FK_Data_2:
      Type = ELF::R_PPC_ADDR16;
      break;
    }
  }
  return Type;
}

unsigned PPCELFObjectWriter::GetRelocType(const MCValue &Target,
                                          const MCFixup &Fixup,
                                          bool IsPCRel,
                                          bool IsRelocWithSymbol,
                                          int64_t Addend) const {
  return getRelocTypeInner(Target, Fixup, IsPCRel);
}

const MCSymbol *PPCELFObjectWriter::undefinedExplicitRelSym(const MCValue &Target,
                                                            const MCFixup &Fixup,
                                                            bool IsPCRel) const {
  assert(Target.getSymA() && "SymA cannot be 0");
  const MCSymbol &Symbol = Target.getSymA()->getSymbol().AliasedSymbol();

  unsigned RelocType = getRelocTypeInner(Target, Fixup, IsPCRel);

  // The .odp creation emits a relocation against the symbol ".TOC." which
  // create a R_PPC64_TOC relocation. However the relocation symbol name
  // in final object creation should be NULL, since the symbol does not
  // really exist, it is just the reference to TOC base for the current
  // object file.
  bool EmitThisSym = RelocType != ELF::R_PPC64_TOC;

  if (EmitThisSym && !Symbol.isTemporary())
    return &Symbol;
  return NULL;
}

MCObjectWriter *llvm::createPPCELFObjectWriter(raw_ostream &OS,
                                               bool Is64Bit,
                                               uint8_t OSABI) {
  MCELFObjectTargetWriter *MOTW = new PPCELFObjectWriter(Is64Bit, OSABI);
  return createELFObjectWriter(MOTW, OS,  /*IsLittleEndian=*/false);
}
