//===-- llvm/CodeGen/AddressPool.h - Dwarf Debug Framework -----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_ASMPRINTER_ADDRESSPOOL_H__
#define CODEGEN_ASMPRINTER_ADDRESSPOOL_H__

#include "llvm/ADT/DenseMap.h"

namespace llvm {
class MCSection;
class MCSymbol;
class AsmPrinter;
// Collection of addresses for this unit and assorted labels.
// A Symbol->unsigned mapping of addresses used by indirect
// references.
class AddressPool {
  struct AddressPoolEntry {
    unsigned Number;
    bool TLS;
    AddressPoolEntry(unsigned Number, bool TLS) : Number(Number), TLS(TLS) {}
  };
  DenseMap<const MCSymbol *, AddressPoolEntry> Pool;
public:
  /// \brief Returns the index into the address pool with the given
  /// label/symbol.
  unsigned getIndex(const MCSymbol *Sym, bool TLS = false);

  void emit(AsmPrinter &Asm, const MCSection *AddrSection);

  bool isEmpty() { return Pool.empty(); }
};
}
#endif
