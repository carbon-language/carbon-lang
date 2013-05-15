//===-- MCELFObjectTargetWriter.cpp - ELF Target Writer Subclass ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"

using namespace llvm;

MCELFObjectTargetWriter::MCELFObjectTargetWriter(bool Is64Bit_,
                                                 uint8_t OSABI_,
                                                 uint16_t EMachine_,
                                                 bool HasRelocationAddend_,
                                                 bool IsN64_)
  : OSABI(OSABI_), EMachine(EMachine_),
    HasRelocationAddend(HasRelocationAddend_), Is64Bit(Is64Bit_),
    IsN64(IsN64_){
}

const MCSymbol *MCELFObjectTargetWriter::ExplicitRelSym(const MCAssembler &Asm,
                                                        const MCValue &Target,
                                                        const MCFragment &F,
                                                        const MCFixup &Fixup,
                                                        bool IsPCRel) const {
  return NULL;
}

const MCSymbol *MCELFObjectTargetWriter::undefinedExplicitRelSym(const MCValue &Target,
                                                                 const MCFixup &Fixup,
                                                                 bool IsPCRel) const {
  const MCSymbol &Symbol = Target.getSymA()->getSymbol();
  return &Symbol.AliasedSymbol();
}

// ELF doesn't require relocations to be in any order. We sort by the r_offset,
// just to match gnu as for easier comparison. The use type and index is an
// arbitrary way of making the sort deterministic.
static int cmpRel(const void *AP, const void *BP) {
  const ELFRelocationEntry &A = *(const ELFRelocationEntry *)AP;
  const ELFRelocationEntry &B = *(const ELFRelocationEntry *)BP;
  if (A.r_offset != B.r_offset)
    return B.r_offset - A.r_offset;
  if (B.Type != A.Type)
    return A.Type - B.Type;
  if (B.Index != A.Index)
    return B.Index - A.Index;
  llvm_unreachable("ELFRelocs might be unstable!");
}

void
MCELFObjectTargetWriter::sortRelocs(const MCAssembler &Asm,
                                    std::vector<ELFRelocationEntry> &Relocs) {
  array_pod_sort(Relocs.begin(), Relocs.end(), cmpRel);
}
