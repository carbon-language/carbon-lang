//===- PDBSymbolTypeVTable.cpp - --------------------------------*- C++ -*-===//
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
#include "llvm/DebugInfo/PDB/PDBSymbolTypeVTable.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeVTableShape.h"

using namespace llvm;

PDBSymbolTypeVTable::PDBSymbolTypeVTable(const IPDBSession &PDBSession,
                                         std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolTypeVTable::dump(raw_ostream &OS, int Indent,
                               PDB_DumpLevel Level) const {
  OS << stream_indent(Indent);
  uint32_t ClassId = getClassParentId();
  if (auto ClassParent = Session.getSymbolById(ClassId)) {
    ClassParent->dump(OS, 0, PDB_DumpLevel::Compact);
    OS << "::";
  }
  OS << "<vtbl> ";
  if (auto VtblPointer =
          Session.getConcreteSymbolById<PDBSymbolTypePointer>(getTypeId())) {
    if (auto VtblShape =
            Session.getConcreteSymbolById<PDBSymbolTypeVTableShape>(
                VtblPointer->getTypeId()))
      OS << "(" << VtblShape->getCount() << " entries)";
  }
  OS.flush();
}
