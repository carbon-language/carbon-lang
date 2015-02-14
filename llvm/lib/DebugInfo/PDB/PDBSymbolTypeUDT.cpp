//===- PDBSymbolTypeUDT.cpp - --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"

#include <utility>

using namespace llvm;

PDBSymbolTypeUDT::PDBSymbolTypeUDT(const IPDBSession &PDBSession,
                                   std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolTypeUDT::dump(raw_ostream &OS, int Indent,
                            PDB_DumpLevel Level) const {
  OS << stream_indent(Indent);
  if (Level >= PDB_DumpLevel::Normal)
    OS << "class ";

  if (isNested()) {
    uint32_t ClassId = getClassParentId();
    if (ClassId != 0) {
      if (auto ClassParent = Session.getSymbolById(ClassId)) {
        ClassParent->dump(OS, 0, Level);
        OS << "::";
      }
    }
  }
  OS << getName();

  if (Level >= PDB_DumpLevel::Normal)
    OS << " (" << getLength() << " bytes)";
}
