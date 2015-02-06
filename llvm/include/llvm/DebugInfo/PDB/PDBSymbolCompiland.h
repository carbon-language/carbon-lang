//===- PDBSymbolCompiland.h - Accessors for querying PDB compilands -----*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLCOMPILAND_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLCOMPILAND_H

#include <string>

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class PDBSymbolCompiland : public PDBSymbol {
public:
  PDBSymbolCompiland(std::unique_ptr<IPDBRawSymbol> CompilandSymbol);

  FORWARD_SYMBOL_METHOD(isEditAndContinueEnabled)
  FORWARD_SYMBOL_METHOD(getLexicalParentId)
  FORWARD_SYMBOL_METHOD(getLibraryName)
  FORWARD_SYMBOL_METHOD(getName)
  FORWARD_SYMBOL_METHOD(getSourceFileName)
  FORWARD_SYMBOL_METHOD(getSymIndexId)

  static bool classof(const PDBSymbol *S) {
    return S->getSymTag() == PDB_SymType::Compiland;
  }
};
}

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLCOMPILAND_H
