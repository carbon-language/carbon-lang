//===-- MCELFObjectTargetWriter.cpp - ELF Target Writer Subclass ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCELFObjectWriter.h"

using namespace llvm;

MCELFObjectTargetWriter::MCELFObjectTargetWriter(bool Is64Bit_,
                                                 uint8_t OSABI_,
                                                 uint16_t EMachine_,
                                                 bool HasRelocationAddend_)
  : OSABI(OSABI_), EMachine(EMachine_),
    HasRelocationAddend(HasRelocationAddend_), Is64Bit(Is64Bit_) {
}

/// Default e_flags = 0
unsigned MCELFObjectTargetWriter::getEFlags() const {
  return 0;
}

const MCSymbol *MCELFObjectTargetWriter::ExplicitRelSym(const MCAssembler &Asm,
                                                        const MCValue &Target,
                                                        const MCFragment &F,
                                                        const MCFixup &Fixup,
                                                        bool IsPCRel) const {
  return NULL;
}


void MCELFObjectTargetWriter::adjustFixupOffset(const MCFixup &Fixup,
                                                uint64_t &RelocOffset) {
}

unsigned MCELFObjectTargetWriter::GetRelocType(const MCValue &Target,
                                               const MCFixup &Fixup,
                                               bool IsPCRel,
                                               bool IsRelocWithSymbol,
                                               int64_t Addend) const {
  return 0;
}

MCELFObjectTargetWriter::~MCELFObjectTargetWriter() {
}
