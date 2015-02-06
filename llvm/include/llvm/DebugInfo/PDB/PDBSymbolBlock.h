//===- PDBSymbolBlock.h - Accessors for querying PDB blocks -------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLBLOCK_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLBLOCK_H

#include <string>

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class PDBSymbolBlock : public PDBSymbol {
public:
  PDBSymbolBlock(std::unique_ptr<IPDBRawSymbol> BlockSymbol);

  FORWARD_SYMBOL_METHOD(getAddressOffset)
  FORWARD_SYMBOL_METHOD(getAddressSection)
  FORWARD_SYMBOL_METHOD(getLength)
  FORWARD_SYMBOL_METHOD(getLexicalParentId)
  FORWARD_SYMBOL_METHOD(getLocationType)
  FORWARD_SYMBOL_METHOD(getName)
  FORWARD_SYMBOL_METHOD(getRelativeVirtualAddress)
  FORWARD_SYMBOL_METHOD(getSymIndexId)
  FORWARD_SYMBOL_METHOD(getVirtualAddress)

  static bool classof(const PDBSymbol *S) {
    return S->getSymTag() == PDB_SymType::Block;
  }
};
}

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLBLOCK_H
