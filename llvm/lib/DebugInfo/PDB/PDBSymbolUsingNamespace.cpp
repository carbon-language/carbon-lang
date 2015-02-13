//===- PDBSymbolUsingNamespace.cpp - ------------------- --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolUsingNamespace.h"
#include <utility>

using namespace llvm;

PDBSymbolUsingNamespace::PDBSymbolUsingNamespace(
    const IPDBSession &PDBSession, std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolUsingNamespace::dump(raw_ostream &OS, int Indent,
                                   PDB_DumpLevel Level) const {}
