//===- PDBSymbolTypeTypedef.cpp - --------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"
#include <utility>
using namespace llvm;

PDBSymbolTypeTypedef::PDBSymbolTypeTypedef(
    const IPDBSession &PDBSession, std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolTypeTypedef::dump(raw_ostream &OS, int Indent,
                                PDB_DumpLevel Level) const {
  OS.indent(Indent);
  if (Level >= PDB_DumpLevel::Normal) {
    std::string Name = getName();
    OS << "typedef:" << Name << " -> ";
    std::string TargetTypeName;
    uint32_t TargetId = getTypeId();
    if (auto TypeSymbol = Session.getSymbolById(TargetId)) {
      TypeSymbol->dump(OS, 0, PDB_DumpLevel::Compact);
    }
    OS << TargetTypeName;
  } else {
    OS << getName();
  }
}
