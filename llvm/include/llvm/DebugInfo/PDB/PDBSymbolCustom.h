//===- PDBSymbolCustom.h - compiler-specific types --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLCUSTOM_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLCUSTOM_H

#include "llvm/ADT/SmallVector.h"

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

/// PDBSymbolCustom represents symbols that are compiler-specific and do not
/// fit anywhere else in the lexical hierarchy.
/// https://msdn.microsoft.com/en-us/library/d88sf09h.aspx
class PDBSymbolCustom : public PDBSymbol {
public:
  PDBSymbolCustom(std::unique_ptr<IPDBRawSymbol> CustomSymbol);

  void getDataBytes(llvm::SmallVector<uint8_t, 32> &bytes);

  FORWARD_SYMBOL_METHOD(getSymIndexId)

  static bool classof(const PDBSymbol *S) {
    return S->getSymTag() == PDB_SymType::Custom;
  }
};

}; // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLCUSTOM_H
