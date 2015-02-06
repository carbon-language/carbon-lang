//===- PDBSymbolExe.h - Accessors for querying executables in a PDB ----*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLEXE_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLEXE_H

#include <string>

#include "llvm/Support/COFF.h"

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class PDBSymbolExe : public PDBSymbol {
public:
  PDBSymbolExe(std::unique_ptr<IPDBRawSymbol> ExeSymbol);

  FORWARD_SYMBOL_METHOD(getAge)
  FORWARD_SYMBOL_METHOD(getGuid)
  FORWARD_SYMBOL_METHOD(hasCTypes)
  FORWARD_SYMBOL_METHOD(hasPrivateSymbols)
  FORWARD_SYMBOL_METHOD(getMachineType)
  FORWARD_SYMBOL_METHOD(getName)
  FORWARD_SYMBOL_METHOD(getSignature)
  FORWARD_SYMBOL_METHOD(getSymbolsFileName)
  FORWARD_SYMBOL_METHOD(getSymIndexId)

  static bool classof(const PDBSymbol *S) {
    return S->getSymTag() == PDB_SymType::Exe;
  }
};
} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLEXE_H
