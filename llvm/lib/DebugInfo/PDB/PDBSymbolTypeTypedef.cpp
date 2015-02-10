//===- PDBSymbolTypeTypedef.cpp - --------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"
using namespace llvm;

PDBSymbolTypeTypedef::PDBSymbolTypeTypedef(
    const IPDBSession &PDBSession, std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolTypeTypedef::dump(raw_ostream &OS, int Indent,
                                PDB_DumpLevel Level) const {
  OS.indent(Indent);
  OS << "typedef:" << getName() << " -> ";
  std::string TargetTypeName;
  auto TypeSymbol = Session.getSymbolById(getTypeId());
  if (PDBSymbolTypeUDT *UDT = dyn_cast<PDBSymbolTypeUDT>(TypeSymbol.get())) {
    TargetTypeName = UDT->getName();
  }
  OS << TargetTypeName << "\n";
}
