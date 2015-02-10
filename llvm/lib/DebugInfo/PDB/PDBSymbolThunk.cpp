//===- PDBSymbolThunk.cpp - -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolThunk.h"

#include "llvm/Support/Format.h"

using namespace llvm;

PDBSymbolThunk::PDBSymbolThunk(const IPDBSession &PDBSession,
                               std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolThunk::dump(raw_ostream &OS, int Indent,
                          PDB_DumpLevel Level) const {
  if (Level == PDB_DumpLevel::Compact) {
    OS.indent(Indent);
    PDB_ThunkOrdinal Ordinal = getThunkOrdinal();
    OS << "THUNK[" << Ordinal << "] ";
    OS << "[" << format_hex(getRelativeVirtualAddress(), 10);
    if (Ordinal == PDB_ThunkOrdinal::TrampIncremental)
      OS << " -> " << format_hex(getTargetRelativeVirtualAddress(), 10);
    OS << "] ";
    std::string Name = getName();
    if (!Name.empty())
      OS << Name;
    OS << "\n";
  }
}
