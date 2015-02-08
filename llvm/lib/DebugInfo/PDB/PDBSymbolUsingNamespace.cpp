//===- PDBSymbolUsingNamespace.cpp - ------------------- --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolUsingNamespace.h"

using namespace llvm;

PDBSymbolUsingNamespace::PDBSymbolUsingNamespace(
    IPDBSession &PDBSession, std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolUsingNamespace::dump(llvm::raw_ostream &OS) const {}
