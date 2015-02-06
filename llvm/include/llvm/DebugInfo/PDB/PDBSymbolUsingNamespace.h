//===- PDBSymbolUsingNamespace.h - using namespace info ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLUSINGNAMESPACE_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLUSINGNAMESPACE_H

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class PDBSymbolUsingNamespace : public PDBSymbol {
public:
  PDBSymbolUsingNamespace(std::unique_ptr<IPDBRawSymbol> UsingSymbol);

  FORWARD_SYMBOL_METHOD(getLexicalParentId)
  FORWARD_SYMBOL_METHOD(getName)
  FORWARD_SYMBOL_METHOD(getSymIndexId)

  static bool classof(const PDBSymbol *S) {
    return S->getSymTag() == PDB_SymType::UsingNamespace;
  }
};

} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLUSINGNAMESPACE_H
