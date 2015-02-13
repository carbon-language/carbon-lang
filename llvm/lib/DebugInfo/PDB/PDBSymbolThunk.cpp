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
  OS.indent(Indent);
  OS << "thunk ";
  PDB_ThunkOrdinal Ordinal = getThunkOrdinal();
  uint32_t RVA = getRelativeVirtualAddress();
  if (Ordinal == PDB_ThunkOrdinal::TrampIncremental) {
    OS << format_hex(RVA, 10);
  } else {
    OS << "[" << format_hex(RVA, 10);
    OS << " - " << format_hex(RVA + getLength(), 10) << "]";
  }
  OS << " (" << Ordinal << ")";
  if (Ordinal == PDB_ThunkOrdinal::TrampIncremental)
    OS << " -> " << format_hex(getTargetRelativeVirtualAddress(), 10);
  OS << " ";
  std::string Name = getName();
  if (!Name.empty())
    OS << Name;
}
