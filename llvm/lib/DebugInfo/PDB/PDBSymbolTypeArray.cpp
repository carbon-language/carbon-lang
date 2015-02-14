//===- PDBSymbolTypeArray.cpp - ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbolTypeArray.h"

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"

#include <utility>

using namespace llvm;

PDBSymbolTypeArray::PDBSymbolTypeArray(const IPDBSession &PDBSession,
                                       std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolTypeArray::dump(raw_ostream &OS, int Indent,
                              PDB_DumpLevel Level) const {
  OS << stream_indent(Indent);
  if (auto ElementType = Session.getSymbolById(getTypeId()))
    ElementType->dump(OS, 0, PDB_DumpLevel::Compact);
  else
    OS << "<unknown-element-type>";
  OS << "[" << getLength() << "]";
}
