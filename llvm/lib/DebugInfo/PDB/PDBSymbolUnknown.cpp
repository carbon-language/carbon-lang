//===- PDBSymbolUnknown.cpp - -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbolUnknown.h"

#include "llvm/DebugInfo/PDB/PDBSymbol.h"

#include <utility>

using namespace llvm;

PDBSymbolUnknown::PDBSymbolUnknown(const IPDBSession &PDBSession,
                                   std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolUnknown::dump(raw_ostream &OS, int Indent,
                            PDB_DumpLevel Level, PDB_DumpFlags Flags) const {}
