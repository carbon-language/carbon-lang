//===- PDBSymbolTypeDimension.h - array dimension type info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEDIMENSION_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEDIMENSION_H

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class PDBSymbolTypeDimension : public PDBSymbol {
public:
  PDBSymbolTypeDimension(std::unique_ptr<IPDBRawSymbol> DimensionTypeSymbol);

  FORWARD_SYMBOL_METHOD(getLowerBoundId)
  FORWARD_SYMBOL_METHOD(getUpperBoundId)
  FORWARD_SYMBOL_METHOD(getSymIndexId)

  static bool classof(const PDBSymbol *S) {
    return S->getSymTag() == PDB_SymType::Dimension;
  }
};

} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEDIMENSION_H
