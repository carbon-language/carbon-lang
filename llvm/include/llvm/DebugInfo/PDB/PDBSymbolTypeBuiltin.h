//===- PDBSymbolTypeBuiltin.h - builtin type information --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEBUILTIN_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEBUILTIN_H

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class PDBSymbolTypeBuiltin : public PDBSymbol {
public:
  PDBSymbolTypeBuiltin(std::unique_ptr<IPDBRawSymbol> BuiltinTypeSymbol);

  FORWARD_SYMBOL_METHOD(getBuiltinType)
  FORWARD_SYMBOL_METHOD(isConstType)
  FORWARD_SYMBOL_METHOD(getLength)
  FORWARD_SYMBOL_METHOD(getLexicalParentId)
  FORWARD_SYMBOL_METHOD(getSymIndexId)
  FORWARD_SYMBOL_METHOD(isUnalignedType)
  FORWARD_SYMBOL_METHOD(isVolatileType)

  static bool classof(const PDBSymbol *S) {
    return S->getSymTag() == PDB_SymType::BuiltinType;
  }
};

} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEBUILTIN_H
