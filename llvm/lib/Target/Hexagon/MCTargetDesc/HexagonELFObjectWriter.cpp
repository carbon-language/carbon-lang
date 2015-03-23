//===-- HexagonELFObjectWriter.cpp - Hexagon Target Descriptions ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
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

  virtual unsigned GetRelocType(MCValue const &Target, MCFixup const &Fixup,
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
  unsigned Type = (unsigned)ELF::R_HEX_NONE;
  llvm::MCFixupKind Kind = Fixup.getKind();

  switch (Kind) {
  default:
    DEBUG(dbgs() << "unrecognized relocation " << Fixup.getKind() << "\n");
    llvm_unreachable("Unimplemented Fixup kind!");
    break;
  case FK_Data_4:
    Type = (IsPCRel) ? ELF::R_HEX_32_PCREL : ELF::R_HEX_32;
    break;
  }
  return Type;
}

MCObjectWriter *llvm::createHexagonELFObjectWriter(raw_ostream &OS,
                                                   uint8_t OSABI,
                                                   StringRef CPU) {
  MCELFObjectTargetWriter *MOTW = new HexagonELFObjectWriter(OSABI, CPU);
  return createELFObjectWriter(MOTW, OS, /*IsLittleEndian*/ true);
}