//===- PDBSymbolCompilandDetails.h - PDB compiland details ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLCOMPILANDDETAILS_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLCOMPILANDDETAILS_H

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class PDBSymbolCompilandDetails : public PDBSymbol {
public:
  PDBSymbolCompilandDetails(std::unique_ptr<IPDBRawSymbol> DetailsSymbol);

  FORWARD_SYMBOL_METHOD(getBackEndBuild)
  FORWARD_SYMBOL_METHOD(getBackEndMajor)
  FORWARD_SYMBOL_METHOD(getBackEndMinor)
  FORWARD_SYMBOL_METHOD(getCompilerName)
  FORWARD_SYMBOL_METHOD(isEditAndContinueEnabled)
  FORWARD_SYMBOL_METHOD(getFrontEndBuild)
  FORWARD_SYMBOL_METHOD(getFrontEndMajor)
  FORWARD_SYMBOL_METHOD(getFrontEndMinor)
  FORWARD_SYMBOL_METHOD(hasDebugInfo)
  FORWARD_SYMBOL_METHOD(hasManagedCode)
  FORWARD_SYMBOL_METHOD(hasSecurityChecks)
  FORWARD_SYMBOL_METHOD(isCVTCIL)
  FORWARD_SYMBOL_METHOD(isDataAligned)
  FORWARD_SYMBOL_METHOD(isHotpatchable)
  FORWARD_SYMBOL_METHOD(isLTCG)
  FORWARD_SYMBOL_METHOD(isMSILNetmodule)
  FORWARD_SYMBOL_METHOD(getLanguage)
  FORWARD_SYMBOL_METHOD(getLexicalParentId)
  FORWARD_SYMBOL_METHOD(getPlatform)
  FORWARD_SYMBOL_METHOD(getSymIndexId)

  static bool classof(const PDBSymbol *S) {
    return S->getSymTag() == PDB_SymType::CompilandDetails;
  }
};

}; // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBFUNCTION_H
