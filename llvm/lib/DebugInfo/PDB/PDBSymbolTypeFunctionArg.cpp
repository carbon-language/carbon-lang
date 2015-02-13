//===- PDBSymbolTypeFunctionArg.cpp - --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionArg.h"
#include <utility>

using namespace llvm;

PDBSymbolTypeFunctionArg::PDBSymbolTypeFunctionArg(
    const IPDBSession &PDBSession, std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolTypeFunctionArg::dump(raw_ostream &OS, int Indent,
                                    PDB_DumpLevel Level) const {
  OS << stream_indent(Indent);
  uint32_t TypeId = getTypeId();
  if (auto Type = Session.getSymbolById(TypeId)) {
    Type->dump(OS, 0, Level);
  }
}
