//===- PDBSymbolTypeCustom.h - custom compiler type information -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPECUSTOM_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPECUSTOM_H

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class PDBSymbolTypeCustom : public PDBSymbol {
public:
  PDBSymbolTypeCustom(std::unique_ptr<IPDBRawSymbol> CustomTypeSymbol);

  FORWARD_SYMBOL_METHOD(getOemId)
  FORWARD_SYMBOL_METHOD(getOemSymbolId)
  FORWARD_SYMBOL_METHOD(getSymIndexId)

  static bool classof(const PDBSymbol *S) {
    return S->getSymTag() == PDB_SymType::CustomType;
  }
};

} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPECUSTOM_H
