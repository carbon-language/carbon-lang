//===- NativeCompilandSymbol.h - Native impl of PDBCompilandSymbol -C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NativeCompilandSymbol.h"

namespace llvm {
namespace pdb {

NativeCompilandSymbol::NativeCompilandSymbol(NativeSession &Session,
                                             const ModuleInfoEx &MI)
    : NativeRawSymbol(Session), Module(MI) {}

PDB_SymType NativeCompilandSymbol::getSymTag() const {
  return PDB_SymType::Compiland;
}

bool NativeCompilandSymbol::isEditAndContinueEnabled() const {
  return Module.Info.hasECInfo();
}

uint32_t NativeCompilandSymbol::getLexicalParentId() const { return 0; }

// DIA, which this API was modeled after, uses counter-intuitive meanings for
// IDiaSymbol::get_name and IDiaSymbol::get_libraryName, which is why these
// methods may appear to be cross-mapped.

std::string NativeCompilandSymbol::getLibraryName() const {
  return Module.Info.getObjFileName();
}

std::string NativeCompilandSymbol::getName() const {
  return Module.Info.getModuleName();
}

} // namespace pdb
} // namespace llvm
