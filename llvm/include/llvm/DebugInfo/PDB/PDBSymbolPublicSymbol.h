//===- PDBSymbolPublicSymbol.h - public symbol info -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLPUBLICSYMBOL_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLPUBLICSYMBOL_H

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class PDBSymbolPublicSymbol : public PDBSymbol {
public:
  PDBSymbolPublicSymbol(std::unique_ptr<IPDBRawSymbol> PublicSymbol);

  FORWARD_SYMBOL_METHOD(getAddressOffset)
  FORWARD_SYMBOL_METHOD(getAddressSection)
  FORWARD_SYMBOL_METHOD(isCode)
  FORWARD_SYMBOL_METHOD(isFunction)
  FORWARD_SYMBOL_METHOD(getLength)
  FORWARD_SYMBOL_METHOD(getLexicalParentId)
  FORWARD_SYMBOL_METHOD(getLocationType)
  FORWARD_SYMBOL_METHOD(isManagedCode)
  FORWARD_SYMBOL_METHOD(isMSILCode)
  FORWARD_SYMBOL_METHOD(getName)
  FORWARD_SYMBOL_METHOD(getSymIndexId)
  FORWARD_SYMBOL_METHOD(getRelativeVirtualAddress)
  FORWARD_SYMBOL_METHOD(getVirtualAddress)
  FORWARD_SYMBOL_METHOD(getUndecoratedName)

  static bool classof(const PDBSymbol *S) {
    return S->getSymTag() == PDB_SymType::PublicSymbol;
  }
};

} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLPUBLICSYMBOL_H
