//===-- MipsELFObjectWriter.cpp - Mips ELF Writer ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MipsFixupKinds.h"
#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
  class MipsELFObjectWriter : public MCELFObjectTargetWriter {
  public:
    MipsELFObjectWriter(uint8_t OSABI);

    virtual ~MipsELFObjectWriter();

    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend) const;
    virtual unsigned getEFlags() const;
    virtual const MCSymbol *ExplicitRelSym(const MCAssembler &Asm,
                                           const MCValue &Target,
                                           const MCFragment &F,
                                           const MCFixup &Fixup,
                                           bool IsPCRel) const;
  };
}

MipsELFObjectWriter::MipsELFObjectWriter(uint8_t OSABI)
  : MCELFObjectTargetWriter(/*Is64Bit*/ false, OSABI, ELF::EM_MIPS,
                            /*HasRelocationAddend*/ false) {}

MipsELFObjectWriter::~MipsELFObjectWriter() {}

// FIXME: get the real EABI Version from the Triple.
unsigned MipsELFObjectWriter::getEFlags() const {
  return ELF::EF_MIPS_NOREORDER | ELF::EF_MIPS_ARCH_32R2;
}

const MCSymbol *MipsELFObjectWriter::ExplicitRelSym(const MCAssembler &Asm,
                                                    const MCValue &Target,
                                                    const MCFragment &F,
                                                    const MCFixup &Fixup,
                                                    bool IsPCRel) const {
  assert(Target.getSymA() && "SymA cannot be 0.");
  const MCSymbol &Sym = Target.getSymA()->getSymbol();

  if (Sym.getSection().getKind().isMergeableCString() ||
      Sym.getSection().getKind().isMergeableConst())
    return &Sym;

  return NULL;
}

unsigned MipsELFObjectWriter::GetRelocType(const MCValue &Target,
                                           const MCFixup &Fixup,
                                           bool IsPCRel,
                                           bool IsRelocWithSymbol,
                                           int64_t Addend) const {
  // determine the type of the relocation
  unsigned Type = (unsigned)ELF::R_MIPS_NONE;
  unsigned Kind = (unsigned)Fixup.getKind();

  switch (Kind) {
  default:
    llvm_unreachable("invalid fixup kind!");
  case FK_Data_4:
    Type = ELF::R_MIPS_32;
    break;
  case FK_GPRel_4:
    Type = ELF::R_MIPS_GPREL32;
    break;
  case Mips::fixup_Mips_GPREL16:
    Type = ELF::R_MIPS_GPREL16;
    break;
  case Mips::fixup_Mips_26:
    Type = ELF::R_MIPS_26;
    break;
  case Mips::fixup_Mips_CALL16:
    Type = ELF::R_MIPS_CALL16;
    break;
  case Mips::fixup_Mips_GOT_Global:
  case Mips::fixup_Mips_GOT_Local:
    Type = ELF::R_MIPS_GOT16;
    break;
  case Mips::fixup_Mips_HI16:
    Type = ELF::R_MIPS_HI16;
    break;
  case Mips::fixup_Mips_LO16:
    Type = ELF::R_MIPS_LO16;
    break;
  case Mips::fixup_Mips_TLSGD:
    Type = ELF::R_MIPS_TLS_GD;
    break;
  case Mips::fixup_Mips_GOTTPREL:
    Type = ELF::R_MIPS_TLS_GOTTPREL;
    break;
  case Mips::fixup_Mips_TPREL_HI:
    Type = ELF::R_MIPS_TLS_TPREL_HI16;
    break;
  case Mips::fixup_Mips_TPREL_LO:
    Type = ELF::R_MIPS_TLS_TPREL_LO16;
    break;
  case Mips::fixup_Mips_TLSLDM:
    Type = ELF::R_MIPS_TLS_LDM;
    break;
  case Mips::fixup_Mips_DTPREL_HI:
    Type = ELF::R_MIPS_TLS_DTPREL_HI16;
    break;
  case Mips::fixup_Mips_DTPREL_LO:
    Type = ELF::R_MIPS_TLS_DTPREL_LO16;
    break;
  case Mips::fixup_Mips_Branch_PCRel:
  case Mips::fixup_Mips_PC16:
    Type = ELF::R_MIPS_PC16;
    break;
  }

  return Type;
}

MCObjectWriter *llvm::createMipsELFObjectWriter(raw_ostream &OS,
                                               bool IsLittleEndian,
                                               uint8_t OSABI) {
  MCELFObjectTargetWriter *MOTW = new MipsELFObjectWriter(OSABI);
  return createELFObjectWriter(MOTW, OS, IsLittleEndian);
}
