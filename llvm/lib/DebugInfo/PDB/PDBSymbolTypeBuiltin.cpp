//===- PDBSymbolTypeBuiltin.cpp - ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBuiltin.h"
#include <utility>

using namespace llvm;

PDBSymbolTypeBuiltin::PDBSymbolTypeBuiltin(
    const IPDBSession &PDBSession, std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolTypeBuiltin::dump(raw_ostream &OS, int Indent,
                                PDB_DumpLevel Level) const {
  OS << stream_indent(Indent);
  PDB_BuiltinType Type = getBuiltinType();
  OS << Type;
  if (Type == PDB_BuiltinType::UInt || Type == PDB_BuiltinType::Int)
    OS << (8 * getLength()) << "_t";
}
