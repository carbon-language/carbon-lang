//===- PDBSymbolTypeUDT.cpp - --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"

using namespace llvm;

PDBSymbolTypeUDT::PDBSymbolTypeUDT(const IPDBSession &PDBSession,
                                   std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolTypeUDT::dump(raw_ostream &OS, int Indent,
                            PDB_DumpLevel Level) const {}
