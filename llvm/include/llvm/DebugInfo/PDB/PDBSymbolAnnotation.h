//===- PDBSymbolAnnotation.h - Accessors for querying PDB annotations ---*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLANNOTATION_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLANNOTATION_H

#include <string>

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class PDBSymbolAnnotation : public PDBSymbol {
public:
  PDBSymbolAnnotation(std::unique_ptr<IPDBRawSymbol> AnnotationSymbol);

  FORWARD_SYMBOL_METHOD(getAddressOffset)
  FORWARD_SYMBOL_METHOD(getAddressSection)
  FORWARD_SYMBOL_METHOD(getDataKind)
  FORWARD_SYMBOL_METHOD(getRelativeVirtualAddress)
  FORWARD_SYMBOL_METHOD(getSymIndexId)
  // FORWARD_SYMBOL_METHOD(getValue)
  FORWARD_SYMBOL_METHOD(getVirtualAddress)

  static bool classof(const PDBSymbol *S) {
    return S->getSymTag() == PDB_SymType::Annotation;
  }
};
}

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLANNOTATION_H
