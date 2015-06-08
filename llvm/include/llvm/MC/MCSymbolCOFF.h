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

  /// This corresponds to the e_type field of the COFF symbol.
  mutable uint16_t Type;

public:
  MCSymbolCOFF(const StringMapEntry<bool> *Name, bool isTemporary)
      : MCSymbol(SymbolKindCOFF, Name, isTemporary), Type(0) {}

  uint16_t getType() const {
    return Type;
  }
  void setType(uint16_t Ty) const {
    Type = Ty;
  }

  static bool classof(const MCSymbol *S) { return S->isCOFF(); }
};
}

#endif
