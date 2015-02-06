//===- PDBSymbolCompilandEnv.h - compiland environment variables *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLCOMPILANDENV_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLCOMPILANDENV_H

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class PDBSymbolCompilandEnv : public PDBSymbol {
public:
  PDBSymbolCompilandEnv(std::unique_ptr<IPDBRawSymbol> CompilandEnvSymbol);

  FORWARD_SYMBOL_METHOD(getLexicalParentId)
  FORWARD_SYMBOL_METHOD(getName)
  FORWARD_SYMBOL_METHOD(getSymIndexId)
  std::string getValue() const;

  static bool classof(const PDBSymbol *S) {
    return S->getSymTag() == PDB_SymType::CompilandEnv;
  }
};

}; // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLCOMPILANDENV_H
