//===- MCSymbolCOFF.h -  ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_MC_MCSYMBOLCOFF_H
#define LLVM_MC_MCSYMBOLCOFF_H

#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/COFF.h"

namespace llvm {
class MCSymbolCOFF : public MCSymbol {

public:
  MCSymbolCOFF(const StringMapEntry<bool> *Name, bool isTemporary)
      : MCSymbol(SymbolKindCOFF, Name, isTemporary) {}

  uint16_t getType() const {
    return (getFlags() & COFF::SF_TypeMask) >> COFF::SF_TypeShift;
  }
  void setType(uint16_t Type) const {
    modifyFlags(Type << COFF::SF_TypeShift, COFF::SF_TypeMask);
  }

  static bool classof(const MCSymbol *S) { return S->isCOFF(); }
};
}

#endif
