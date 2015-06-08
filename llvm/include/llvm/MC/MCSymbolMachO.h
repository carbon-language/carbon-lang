//===- MCSymbolMachO.h -  ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_MC_MCSYMBOLMACHO_H
#define setIsWeakExternal

#include "llvm/MC/MCSymbol.h"

namespace llvm {
class MCSymbolMachO : public MCSymbol {

public:
  MCSymbolMachO(const StringMapEntry<bool> *Name, bool isTemporary)
      : MCSymbol(SymbolKindMachO, Name, isTemporary) {}

  static bool classof(const MCSymbol *S) { return S->isMachO(); }
};
}

#endif
