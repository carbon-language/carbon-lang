//===-- HexagonELFObjectWriter.cpp - Hexagon Target Descriptions ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "MCTargetDesc/HexagonFixupKinds.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "hexagon-elf-writer"

using namespace llvm;
using namespace Hexagon;

namespace {

class HexagonELFObjectWriter : public MCELFObjectTargetWriter {
private:
  StringRef CPU;

public:
  HexagonELFObjectWriter(uint8_t OSABI, StringRef C);

  unsigned GetRelocType(MCValue const &Target, MCFixup const &Fixup,
                        bool IsPCRel) const override;
};
}

HexagonELFObjectWriter::HexagonELFObjectWriter(uint8_t OSABI, StringRef C)
    : MCELFObjectTargetWriter(/*Is64bit*/ false, OSABI, ELF::EM_HEXAGON,
                              /*HasRelocationAddend*/ true),
      CPU(C) {}

unsigned HexagonELFObjectWriter::GetRelocType(MCValue const &/*Target*/,
                                              MCFixup const &Fixup,
                                              bool IsPCRel) const {
  // determine the type of the relocation
  unsigned Type = (unsigned)ELF::R_HEX_NONE;
  unsigned Kind = (unsigned)Fixup.getKind();

  switch (Kind) {
    default:
      DEBUG(dbgs() << "unrecognized relocation " << Fixup.getKind() << "\n");
      llvm_unreachable("Unimplemented Fixup kind!");
      break;
    case FK_Data_4:
      Type = (IsPCRel) ? ELF::R_HEX_32_PCREL : ELF::R_HEX_32;
      break;
    case FK_PCRel_4:
      Type = ELF::R_HEX_32_PCREL;
      break;
    case FK_Data_2:
      Type = ELF::R_HEX_16;
      break;
   case FK_Data_1:
      Type = ELF::R_HEX_8;
      break;
    case fixup_Hexagon_B22_PCREL:
      Type = ELF::R_HEX_B22_PCREL;
      break;
    case fixup_Hexagon_B15_PCREL:
      Type = ELF::R_HEX_B15_PCREL;
      break;
    case fixup_Hexagon_B7_PCREL:
      Type = ELF::R_HEX_B7_PCREL;
      break;
    case fixup_Hexagon_LO16:
      Type = ELF::R_HEX_LO16;
      break;
    case fixup_Hexagon_HI16:
      Type = ELF::R_HEX_HI16;
      break;
    case fixup_Hexagon_32:
      Type = ELF::R_HEX_32;
      break;
    case fixup_Hexagon_16:
      Type = ELF::R_HEX_16;
      break;
    case fixup_Hexagon_8:
      Type = ELF::R_HEX_8;
      break;
    case fixup_Hexagon_GPREL16_0:
      Type = ELF::R_HEX_GPREL16_0;
      break;
    case fixup_Hexagon_GPREL16_1:
      Type = ELF::R_HEX_GPREL16_1;
      break;
    case fixup_Hexagon_GPREL16_2:
      Type = ELF::R_HEX_GPREL16_2;
      break;
    case fixup_Hexagon_GPREL16_3:
      Type = ELF::R_HEX_GPREL16_3;
      break;
    case fixup_Hexagon_HL16:
      Type = ELF::R_HEX_HL16;
      break;
    case fixup_Hexagon_B13_PCREL:
      Type = ELF::R_HEX_B13_PCREL;
      break;
    case fixup_Hexagon_B9_PCREL:
      Type = ELF::R_HEX_B9_PCREL;
      break;
    case fixup_Hexagon_B32_PCREL_X:
      Type = ELF::R_HEX_B32_PCREL_X;
      break;
    case fixup_Hexagon_32_6_X:
      Type = ELF::R_HEX_32_6_X;
      break;
    case fixup_Hexagon_B22_PCREL_X:
      Type = ELF::R_HEX_B22_PCREL_X;
      break;
    case fixup_Hexagon_B15_PCREL_X:
      Type = ELF::R_HEX_B15_PCREL_X;
      break;
    case fixup_Hexagon_B13_PCREL_X:
      Type = ELF::R_HEX_B13_PCREL_X;
      break;
    case fixup_Hexagon_B9_PCREL_X:
      Type = ELF::R_HEX_B9_PCREL_X;
      break;
    case fixup_Hexagon_B7_PCREL_X:
      Type = ELF::R_HEX_B7_PCREL_X;
      break;
    case fixup_Hexagon_16_X:
      Type = ELF::R_HEX_16_X;
      break;
    case fixup_Hexagon_12_X:
      Type = ELF::R_HEX_12_X;
      break;
    case fixup_Hexagon_11_X:
      Type = ELF::R_HEX_11_X;
      break;
    case fixup_Hexagon_10_X:
      Type = ELF::R_HEX_10_X;
      break;
    case fixup_Hexagon_9_X:
      Type = ELF::R_HEX_9_X;
      break;
    case fixup_Hexagon_8_X:
      Type = ELF::R_HEX_8_X;
      break;
    case fixup_Hexagon_7_X:
      Type = ELF::R_HEX_7_X;
      break;
    case fixup_Hexagon_6_X:
      Type = ELF::R_HEX_6_X;
      break;
    case fixup_Hexagon_32_PCREL:
      Type = ELF::R_HEX_32_PCREL;
      break;
    case fixup_Hexagon_COPY:
      Type = ELF::R_HEX_COPY;
      break;
    case fixup_Hexagon_GLOB_DAT:
      Type = ELF::R_HEX_GLOB_DAT;
      break;
    case fixup_Hexagon_JMP_SLOT:
      Type = ELF::R_HEX_JMP_SLOT;
      break;
    case fixup_Hexagon_RELATIVE:
      Type = ELF::R_HEX_RELATIVE;
      break;
    case fixup_Hexagon_PLT_B22_PCREL:
      Type = ELF::R_HEX_PLT_B22_PCREL;
      break;
    case fixup_Hexagon_GOTREL_LO16:
      Type = ELF::R_HEX_GOTREL_LO16;
      break;
    case fixup_Hexagon_GOTREL_HI16:
      Type = ELF::R_HEX_GOTREL_HI16;
      break;
    case fixup_Hexagon_GOTREL_32:
      Type = ELF::R_HEX_GOTREL_32;
      break;
    case fixup_Hexagon_GOT_LO16:
      Type = ELF::R_HEX_GOT_LO16;
      break;
    case fixup_Hexagon_GOT_HI16:
      Type = ELF::R_HEX_GOT_HI16;
      break;
    case fixup_Hexagon_GOT_32:
      Type = ELF::R_HEX_GOT_32;
      break;
    case fixup_Hexagon_GOT_16:
      Type = ELF::R_HEX_GOT_16;
      break;
    case fixup_Hexagon_DTPMOD_32:
      Type = ELF::R_HEX_DTPMOD_32;
      break;
    case fixup_Hexagon_DTPREL_LO16:
      Type = ELF::R_HEX_DTPREL_LO16;
      break;
    case fixup_Hexagon_DTPREL_HI16:
      Type = ELF::R_HEX_DTPREL_HI16;
      break;
    case fixup_Hexagon_DTPREL_32:
      Type = ELF::R_HEX_DTPREL_32;
      break;
    case fixup_Hexagon_DTPREL_16:
      Type = ELF::R_HEX_DTPREL_16;
      break;
    case fixup_Hexagon_GD_PLT_B22_PCREL:
      Type = ELF::R_HEX_GD_PLT_B22_PCREL;
      break;
    case fixup_Hexagon_LD_PLT_B22_PCREL:
      Type = ELF::R_HEX_LD_PLT_B22_PCREL;
      break;
    case fixup_Hexagon_GD_GOT_LO16:
      Type = ELF::R_HEX_GD_GOT_LO16;
      break;
    case fixup_Hexagon_GD_GOT_HI16:
      Type = ELF::R_HEX_GD_GOT_HI16;
      break;
    case fixup_Hexagon_GD_GOT_32:
      Type = ELF::R_HEX_GD_GOT_32;
      break;
    case fixup_Hexagon_GD_GOT_16:
      Type = ELF::R_HEX_GD_GOT_16;
      break;
    case fixup_Hexagon_LD_GOT_LO16:
      Type = ELF::R_HEX_LD_GOT_LO16;
      break;
    case fixup_Hexagon_LD_GOT_HI16:
      Type = ELF::R_HEX_LD_GOT_HI16;
      break;
    case fixup_Hexagon_LD_GOT_32:
      Type = ELF::R_HEX_LD_GOT_32;
      break;
    case fixup_Hexagon_LD_GOT_16:
      Type = ELF::R_HEX_LD_GOT_16;
      break;
    case fixup_Hexagon_IE_LO16:
      Type = ELF::R_HEX_IE_LO16;
      break;
    case fixup_Hexagon_IE_HI16:
      Type = ELF::R_HEX_IE_HI16;
      break;
    case fixup_Hexagon_IE_32:
      Type = ELF::R_HEX_IE_32;
      break;
    case fixup_Hexagon_IE_GOT_LO16:
      Type = ELF::R_HEX_IE_GOT_LO16;
      break;
    case fixup_Hexagon_IE_GOT_HI16:
      Type = ELF::R_HEX_IE_GOT_HI16;
      break;
    case fixup_Hexagon_IE_GOT_32:
      Type = ELF::R_HEX_IE_GOT_32;
      break;
    case fixup_Hexagon_IE_GOT_16:
      Type = ELF::R_HEX_IE_GOT_16;
      break;
    case fixup_Hexagon_TPREL_LO16:
      Type = ELF::R_HEX_TPREL_LO16;
      break;
    case fixup_Hexagon_TPREL_HI16:
      Type = ELF::R_HEX_TPREL_HI16;
      break;
    case fixup_Hexagon_TPREL_32:
      Type = ELF::R_HEX_TPREL_32;
      break;
    case fixup_Hexagon_TPREL_16:
      Type = ELF::R_HEX_TPREL_16;
      break;
    case fixup_Hexagon_6_PCREL_X:
      Type = ELF::R_HEX_6_PCREL_X;
      break;
    case fixup_Hexagon_GOTREL_32_6_X:
      Type = ELF::R_HEX_GOTREL_32_6_X;
      break;
    case fixup_Hexagon_GOTREL_16_X:
      Type = ELF::R_HEX_GOTREL_16_X;
      break;
    case fixup_Hexagon_GOTREL_11_X:
      Type = ELF::R_HEX_GOTREL_11_X;
      break;
    case fixup_Hexagon_GOT_32_6_X:
      Type = ELF::R_HEX_GOT_32_6_X;
      break;
    case fixup_Hexagon_GOT_16_X:
      Type = ELF::R_HEX_GOT_16_X;
      break;
    case fixup_Hexagon_GOT_11_X:
      Type = ELF::R_HEX_GOT_11_X;
      break;
    case fixup_Hexagon_DTPREL_32_6_X:
      Type = ELF::R_HEX_DTPREL_32_6_X;
      break;
    case fixup_Hexagon_DTPREL_16_X:
      Type = ELF::R_HEX_DTPREL_16_X;
      break;
    case fixup_Hexagon_DTPREL_11_X:
      Type = ELF::R_HEX_DTPREL_11_X;
      break;
    case fixup_Hexagon_GD_GOT_32_6_X:
      Type = ELF::R_HEX_GD_GOT_32_6_X;
      break;
    case fixup_Hexagon_GD_GOT_16_X:
      Type = ELF::R_HEX_GD_GOT_16_X;
      break;
    case fixup_Hexagon_GD_GOT_11_X:
      Type = ELF::R_HEX_GD_GOT_11_X;
      break;
    case fixup_Hexagon_LD_GOT_32_6_X:
      Type = ELF::R_HEX_LD_GOT_32_6_X;
      break;
    case fixup_Hexagon_LD_GOT_16_X:
      Type = ELF::R_HEX_LD_GOT_16_X;
      break;
    case fixup_Hexagon_LD_GOT_11_X:
      Type = ELF::R_HEX_LD_GOT_11_X;
      break;
    case fixup_Hexagon_IE_32_6_X:
      Type = ELF::R_HEX_IE_32_6_X;
      break;
    case fixup_Hexagon_IE_16_X:
      Type = ELF::R_HEX_IE_16_X;
      break;
    case fixup_Hexagon_IE_GOT_32_6_X:
      Type = ELF::R_HEX_IE_GOT_32_6_X;
      break;
    case fixup_Hexagon_IE_GOT_16_X:
      Type = ELF::R_HEX_IE_GOT_16_X;
      break;
    case fixup_Hexagon_IE_GOT_11_X:
      Type = ELF::R_HEX_IE_GOT_11_X;
      break;
    case fixup_Hexagon_TPREL_32_6_X:
      Type = ELF::R_HEX_TPREL_32_6_X;
      break;
    case fixup_Hexagon_TPREL_16_X:
      Type = ELF::R_HEX_TPREL_16_X;
      break;
    case fixup_Hexagon_TPREL_11_X:
      Type = ELF::R_HEX_TPREL_11_X;
      break;
  }
  return Type;
}

MCObjectWriter *llvm::createHexagonELFObjectWriter(raw_pwrite_stream &OS,
                                                   uint8_t OSABI,
                                                   StringRef CPU) {
  MCELFObjectTargetWriter *MOTW = new HexagonELFObjectWriter(OSABI, CPU);
  return createELFObjectWriter(MOTW, OS, /*IsLittleEndian*/ true);
}
