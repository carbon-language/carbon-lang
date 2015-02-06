//===- PDBSymbolUnknown.h - unknown symbol type -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLUNKNOWN_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLUNKNOWN_H

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class PDBSymbolUnknown : public PDBSymbol {
public:
  PDBSymbolUnknown(std::unique_ptr<IPDBRawSymbol> UnknownSymbol);

  static bool classof(const PDBSymbol *S) {
    return (S->getSymTag() == PDB_SymType::None ||
            S->getSymTag() >= PDB_SymType::Max);
  }
};

} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLUNKNOWN_H
