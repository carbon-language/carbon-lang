//===-- X86ELFObjectWriter.cpp - X86 ELF Writer ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86FixupKinds.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
  class X86ELFObjectWriter : public MCELFObjectTargetWriter {
  public:
    X86ELFObjectWriter(bool IsELF64, uint8_t OSABI, uint16_t EMachine);

    virtual ~X86ELFObjectWriter();
  protected:
    unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                          bool IsPCRel) const override;
  };
}

X86ELFObjectWriter::X86ELFObjectWriter(bool IsELF64, uint8_t OSABI,
                                       uint16_t EMachine)
  : MCELFObjectTargetWriter(IsELF64, OSABI, EMachine,
                            // Only i386 uses Rel instead of RelA.
                            /*HasRelocationAddend*/ EMachine != ELF::EM_386) {}

X86ELFObjectWriter::~X86ELFObjectWriter()
{}

unsigned X86ELFObjectWriter::GetRelocType(const MCValue &Target,
                                          const MCFixup &Fixup,
                                          bool IsPCRel) const {
  // determine the type of the relocation

  MCSymbolRefExpr::VariantKind Modifier = Target.getAccessVariant();
  if (getEMachine() == ELF::EM_X86_64) {
    if (IsPCRel) {
      switch ((unsigned)Fixup.getKind()) {
      default:
        llvm_unreachable("invalid fixup kind!");

      case FK_Data_8:
        return ELF::R_X86_64_PC64;
      case FK_Data_4:
        return ELF::R_X86_64_PC32;
      case FK_Data_2:
        return ELF::R_X86_64_PC16;
      case FK_Data_1:
        return ELF::R_X86_64_PC8;
      case FK_PCRel_8:
        assert(Modifier == MCSymbolRefExpr::VK_None);
        return ELF::R_X86_64_PC64;
      case X86::reloc_signed_4byte:
      case X86::reloc_riprel_4byte_movq_load:
      case X86::reloc_riprel_4byte:
      case FK_PCRel_4:
        switch (Modifier) {
        default:
          llvm_unreachable("Unimplemented");
        case MCSymbolRefExpr::VK_None:
          return ELF::R_X86_64_PC32;
        case MCSymbolRefExpr::VK_PLT:
          return ELF::R_X86_64_PLT32;
        case MCSymbolRefExpr::VK_GOTPCREL:
          return ELF::R_X86_64_GOTPCREL;
        case MCSymbolRefExpr::VK_GOTTPOFF:
          return ELF::R_X86_64_GOTTPOFF;
        case MCSymbolRefExpr::VK_TLSGD:
          return ELF::R_X86_64_TLSGD;
        case MCSymbolRefExpr::VK_TLSLD:
          return ELF::R_X86_64_TLSLD;
        }
      case FK_PCRel_2:
        assert(Modifier == MCSymbolRefExpr::VK_None);
        return ELF::R_X86_64_PC16;
      case FK_PCRel_1:
        assert(Modifier == MCSymbolRefExpr::VK_None);
        return ELF::R_X86_64_PC8;
      }
    }
    switch ((unsigned)Fixup.getKind()) {
    default:
      llvm_unreachable("invalid fixup kind!");
    case X86::reloc_global_offset_table8:
      return ELF::R_X86_64_GOTPC64;
    case X86::reloc_global_offset_table:
      return ELF::R_X86_64_GOTPC32;
    case FK_Data_8:
      switch (Modifier) {
      default:
        llvm_unreachable("Unimplemented");
      case MCSymbolRefExpr::VK_None:
        return ELF::R_X86_64_64;
      case MCSymbolRefExpr::VK_GOT:
        return ELF::R_X86_64_GOT64;
      case MCSymbolRefExpr::VK_GOTOFF:
        return ELF::R_X86_64_GOTOFF64;
      case MCSymbolRefExpr::VK_TPOFF:
        return ELF::R_X86_64_TPOFF64;
      case MCSymbolRefExpr::VK_DTPOFF:
        return ELF::R_X86_64_DTPOFF64;
      case MCSymbolRefExpr::VK_SIZE:
        return ELF::R_X86_64_SIZE64;
      }
    case X86::reloc_signed_4byte:
      switch (Modifier) {
      default:
        llvm_unreachable("Unimplemented");
      case MCSymbolRefExpr::VK_None:
        return ELF::R_X86_64_32S;
      case MCSymbolRefExpr::VK_GOT:
        return ELF::R_X86_64_GOT32;
      case MCSymbolRefExpr::VK_GOTPCREL:
        return ELF::R_X86_64_GOTPCREL;
      case MCSymbolRefExpr::VK_TPOFF:
        return ELF::R_X86_64_TPOFF32;
      case MCSymbolRefExpr::VK_DTPOFF:
        return ELF::R_X86_64_DTPOFF32;
      case MCSymbolRefExpr::VK_SIZE:
        return ELF::R_X86_64_SIZE32;
      }
    case FK_Data_4:
      return ELF::R_X86_64_32;
    case FK_Data_2:
      return ELF::R_X86_64_16;
    case FK_PCRel_1:
    case FK_Data_1:
      return ELF::R_X86_64_8;
    }
  }
  assert(getEMachine() == ELF::EM_386 && "Unsupported ELF machine type.");
  if (IsPCRel) {
    switch ((unsigned)Fixup.getKind()) {
    default:
      llvm_unreachable("invalid fixup kind!");

    case X86::reloc_global_offset_table:
      return ELF::R_386_GOTPC;
    case FK_PCRel_1:
    case FK_Data_1:
      switch (Modifier) {
      default:
        llvm_unreachable("Unimplemented");
      case MCSymbolRefExpr::VK_None:
        return ELF::R_386_PC8;
      }
    case FK_PCRel_2:
    case FK_Data_2:
      switch (Modifier) {
      default:
        llvm_unreachable("Unimplemented");
      case MCSymbolRefExpr::VK_None:
        return ELF::R_386_PC16;
      }
    case X86::reloc_riprel_4byte:
    case X86::reloc_signed_4byte:
    case FK_PCRel_4:
    case FK_Data_4:
      switch (Modifier) {
      default:
        llvm_unreachable("Unimplemented");
      case MCSymbolRefExpr::VK_None:
        return ELF::R_386_PC32;
      case MCSymbolRefExpr::VK_PLT:
        return ELF::R_386_PLT32;
      }
    }
  } else {
    switch ((unsigned)Fixup.getKind()) {
    default:
      llvm_unreachable("invalid fixup kind!");
    case X86::reloc_global_offset_table:
      return ELF::R_386_GOTPC;

    // FIXME: Should we avoid selecting reloc_signed_4byte in 32 bit mode
    // instead?
    case X86::reloc_signed_4byte:
    case FK_PCRel_4:
    case FK_Data_4:
      switch (Modifier) {
      default:
        llvm_unreachable("Unimplemented");
      case MCSymbolRefExpr::VK_None:
        return ELF::R_386_32;
      case MCSymbolRefExpr::VK_GOT:
        return ELF::R_386_GOT32;
      case MCSymbolRefExpr::VK_PLT:
        return ELF::R_386_PLT32;
      case MCSymbolRefExpr::VK_GOTOFF:
        return ELF::R_386_GOTOFF;
      case MCSymbolRefExpr::VK_TLSGD:
        return ELF::R_386_TLS_GD;
      case MCSymbolRefExpr::VK_TPOFF:
        return ELF::R_386_TLS_LE_32;
      case MCSymbolRefExpr::VK_INDNTPOFF:
        return ELF::R_386_TLS_IE;
      case MCSymbolRefExpr::VK_NTPOFF:
        return ELF::R_386_TLS_LE;
      case MCSymbolRefExpr::VK_GOTNTPOFF:
        return ELF::R_386_TLS_GOTIE;
      case MCSymbolRefExpr::VK_TLSLDM:
        return ELF::R_386_TLS_LDM;
      case MCSymbolRefExpr::VK_DTPOFF:
        return ELF::R_386_TLS_LDO_32;
      case MCSymbolRefExpr::VK_GOTTPOFF:
        return ELF::R_386_TLS_IE_32;
      }
    case FK_Data_2:
      return ELF::R_386_16;
    case FK_PCRel_1:
    case FK_Data_1:
      return ELF::R_386_8;
    }
  }
}

MCObjectWriter *llvm::createX86ELFObjectWriter(raw_ostream &OS,
                                               bool IsELF64,
                                               uint8_t OSABI,
                                               uint16_t EMachine) {
  MCELFObjectTargetWriter *MOTW =
    new X86ELFObjectWriter(IsELF64, OSABI, EMachine);
  return createELFObjectWriter(MOTW, OS,  /*IsLittleEndian=*/true);
}
