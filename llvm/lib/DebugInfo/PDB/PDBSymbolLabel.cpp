//===- PDBSymbolLabel.cpp - -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolLabel.h"

#include "llvm/Support/Format.h"

using namespace llvm;

PDBSymbolLabel::PDBSymbolLabel(const IPDBSession &PDBSession,
                               std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolLabel::dump(raw_ostream &OS, int Indent,
                          PDB_DumpLevel Level) const {
  OS << stream_indent(Indent);
  OS << "label [" << format_hex(getRelativeVirtualAddress(), 10) << "] "
     << getName();
}
