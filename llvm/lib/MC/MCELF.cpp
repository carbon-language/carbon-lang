//===- lib/MC/MCELF.cpp - MC ELF ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF object file writer information.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCELF.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCELFSymbolFlags.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/Support/ELF.h"

namespace llvm {

void MCELF::SetBinding(const MCSymbol &Sym, unsigned Binding) {
  assert(Binding == ELF::STB_LOCAL || Binding == ELF::STB_GLOBAL ||
         Binding == ELF::STB_WEAK || Binding == ELF::STB_GNU_UNIQUE);
  uint32_t OtherFlags = Sym.getFlags() & ~(0xf << ELF_STB_Shift);
  Sym.setFlags(OtherFlags | (Binding << ELF_STB_Shift));
}

unsigned MCELF::GetBinding(const MCSymbol &Sym) {
  uint32_t Binding = (Sym.getFlags() & (0xf << ELF_STB_Shift)) >> ELF_STB_Shift;
  assert(Binding == ELF::STB_LOCAL || Binding == ELF::STB_GLOBAL ||
         Binding == ELF::STB_WEAK || Binding == ELF::STB_GNU_UNIQUE);
  return Binding;
}

void MCELF::SetType(const MCSymbol &Sym, unsigned Type) {
  assert(Type == ELF::STT_NOTYPE || Type == ELF::STT_OBJECT ||
         Type == ELF::STT_FUNC || Type == ELF::STT_SECTION ||
         Type == ELF::STT_COMMON || Type == ELF::STT_TLS ||
         Type == ELF::STT_GNU_IFUNC);

  uint32_t OtherFlags = Sym.getFlags() & ~(0xf << ELF_STT_Shift);
  Sym.setFlags(OtherFlags | (Type << ELF_STT_Shift));
}

unsigned MCELF::GetType(const MCSymbol &Sym) {
  uint32_t Type = (Sym.getFlags() & (0xf << ELF_STT_Shift)) >> ELF_STT_Shift;
  assert(Type == ELF::STT_NOTYPE || Type == ELF::STT_OBJECT ||
         Type == ELF::STT_FUNC || Type == ELF::STT_SECTION ||
         Type == ELF::STT_COMMON || Type == ELF::STT_TLS || Type == ELF::STT_GNU_IFUNC);
  return Type;
}

// Visibility is stored in the first two bits of st_other
// st_other values are stored in the second byte of get/setFlags
void MCELF::SetVisibility(MCSymbol &Sym, unsigned Visibility) {
  assert(Visibility == ELF::STV_DEFAULT || Visibility == ELF::STV_INTERNAL ||
         Visibility == ELF::STV_HIDDEN || Visibility == ELF::STV_PROTECTED);

  uint32_t OtherFlags = Sym.getFlags() & ~(0x3 << ELF_STV_Shift);
  Sym.setFlags(OtherFlags | (Visibility << ELF_STV_Shift));
}

unsigned MCELF::GetVisibility(const MCSymbol &Sym) {
  unsigned Visibility =
      (Sym.getFlags() & (0x3 << ELF_STV_Shift)) >> ELF_STV_Shift;
  assert(Visibility == ELF::STV_DEFAULT || Visibility == ELF::STV_INTERNAL ||
         Visibility == ELF::STV_HIDDEN || Visibility == ELF::STV_PROTECTED);
  return Visibility;
}

// Other is stored in the last six bits of st_other
// st_other values are stored in the second byte of get/setFlags
void MCELF::setOther(MCSymbol &Sym, unsigned Other) {
  uint32_t OtherFlags = Sym.getFlags() & ~(0x3f << ELF_STO_Shift);
  Sym.setFlags(OtherFlags | (Other << ELF_STO_Shift));
}

unsigned MCELF::getOther(const MCSymbol &Sym) {
  unsigned Other = (Sym.getFlags() & (0x3f << ELF_STO_Shift)) >> ELF_STO_Shift;
  return Other;
}

}
