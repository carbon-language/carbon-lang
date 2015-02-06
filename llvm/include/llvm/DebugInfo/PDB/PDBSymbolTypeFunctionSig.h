//===- PDBSymbolTypeFunctionSig.h - function signature type info *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEFUNCTIONSIG_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEFUNCTIONSIG_H

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class PDBSymbolTypeFunctionSig : public PDBSymbol {
public:
  PDBSymbolTypeFunctionSig(std::unique_ptr<IPDBRawSymbol> FuncSigTypeSymbol);

  FORWARD_SYMBOL_METHOD(getCallingConvention)
  FORWARD_SYMBOL_METHOD(getClassParentId)
  FORWARD_SYMBOL_METHOD(isConstType)
  FORWARD_SYMBOL_METHOD(getCount)
  FORWARD_SYMBOL_METHOD(getLexicalParentId)
  // FORWARD_SYMBOL_METHOD(getObjectPointerType)
  FORWARD_SYMBOL_METHOD(getSymIndexId)
  FORWARD_SYMBOL_METHOD(getThisAdjust)
  FORWARD_SYMBOL_METHOD(getTypeId)
  FORWARD_SYMBOL_METHOD(isUnalignedType)
  FORWARD_SYMBOL_METHOD(isVolatileType)

  static bool classof(const PDBSymbol *S) {
    return S->getSymTag() == PDB_SymType::FunctionSig;
  }
};

} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEFUNCTIONSIG_H
