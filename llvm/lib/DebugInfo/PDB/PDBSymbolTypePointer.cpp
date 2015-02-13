//===- PDBSymbolTypePointer.cpp - --------------------------------*- C++
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
#include "llvm/DebugInfo/PDB/PDBSymbolTypePointer.h"

using namespace llvm;

PDBSymbolTypePointer::PDBSymbolTypePointer(
    const IPDBSession &PDBSession, std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolTypePointer::dump(raw_ostream &OS, int Indent,
                                PDB_DumpLevel Level) const {
  OS << stream_indent(Indent);
  if (isConstType())
    OS << "const ";
  if (isVolatileType())
    OS << "volatile ";
  uint32_t PointeeId = getTypeId();
  if (auto PointeeType = Session.getSymbolById(PointeeId))
    PointeeType->dump(OS, 0, PDB_DumpLevel::Compact);
  OS << ((isReference()) ? "&" : "*");
}
